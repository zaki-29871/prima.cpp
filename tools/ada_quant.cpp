#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "llama.h"
#include "profiler.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

// ============================================================
// Adaptive Quantization Offline Scheduling Tool
// ============================================================
// This tool performs offline optimization for distributed inference
// by solving the layer allocation and bitwidth selection problem
// using HiGHS MILP solver.
//
// Input:
//   - Model statistics from CSV (Model_Statistics.csv)
//   - Device performance data (from profiling)
//   - Number of devices and their configurations
//
// Output:
//   - Layer allocation per device
//   - Optimal bitwidth per device (CPU/GPU)
//   - Detailed scheduling table
// ============================================================

static const char* MODEL_STATS_CSV_DEFAULT = "/root/prima.cpp/Model_Statistics.csv";

// Print usage information
static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\n");
    LOG("Adaptive Quantization Offline Scheduling Tool\n");
    LOG("Usage:\n");
    LOG("  Single device profiling:\n");
    LOG("    %s -m model.gguf --ada-quant\n", argv[0]);
    LOG("\n");
    LOG("  Multi-device offline scheduling:\n");
    LOG("    %s -m model.gguf --ada-quant --world N --rank R --master IP --next IP\n", argv[0]);
    LOG("\n");
    LOG("Options:\n");
    LOG("  -m FNAME          Model file (default fp16 model used for structure only)\n");
    LOG("  --world N         Number of devices in the cluster\n");
    LOG("  --rank R          Rank of this device (0 for master)\n");
    LOG("  --master IP       IP address of master node\n");
    LOG("  --next IP         IP address of next node in ring\n");
    LOG("  --ada-quant       Enable adaptive quantization mode\n");
    LOG("  --n_ctx N         Context size (default: 2048)\n");
    LOG("  --gpu-mem N       GPU memory to use in GiB (default: 999)\n");
    LOG("  --n-threads N     Number of CPU threads (default: -1 for auto)\n");
    LOG("  --model-stats PATH Path to Model_Statistics.csv\n");
    LOG("  --quality-budget MAX Quality budget (default: 1e30, effectively unlimited)\n");
    LOG("\n");
    LOG("Note: This is an offline tool - it solves the optimization problem\n");
    LOG("      and outputs the scheduling result without running inference.\n");
    LOG("\n");
}

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    // ============================================================
    // 快速敏感度测试（如果提供了 --test-sensitivity 参数）
    // ============================================================
    // 使用方法: ./llama-ada-quant --ada-quant --test-sensitivity
    if (params.test_sensitivity) {
        LOG_INF("\n=== Sensitivity Test Mode ===\n");

        bitwidth_model_params test_params;

        const char* variance_csv = "/root/prima.cpp/tensor_capture_stats.csv";
        const char* minmax_csv = "/root/prima.cpp/tensor_minmax_stats_all.csv";

        if (!test_params.load_sensitivity_data(variance_csv, minmax_csv)) {
            LOG_ERR("Failed to load sensitivity data\n");
            return 1;
        }

        LOG_INF("\n=== Layer 0 Sensitivity at Different Bitwidths ===\n");
        LOG_INF("Bitwidth | Sensitivity    | Scale (relative to 1-bit)\n");
        LOG_INF("---------|----------------|-------------------------\n");

        float baseline_loss = 0.0f;
        for (int bw : {1, 2, 4, 8, 16}) {
            float loss = test_params.get_quality_loss(0, bw);
            if (bw == 1) baseline_loss = loss;
            float scale = (baseline_loss > 0) ? (loss / baseline_loss) : 0.0f;
            LOG_INF("%8d | %14.6f | %6.4f%%\n", bw, loss, scale * 100.0f);
        }

        LOG_INF("\n=== Validation ===\n");
        float loss_1bit = test_params.get_quality_loss(0, 1);
        float loss_4bit = test_params.get_quality_loss(0, 4);
        float loss_16bit = test_params.get_quality_loss(0, 16);

        LOG_INF("Layer 0, bitwidth 1:  %f\n", loss_1bit);
        LOG_INF("Layer 0, bitwidth 4:  %f\n", loss_4bit);
        LOG_INF("Layer 0, bitwidth 16: %f\n", loss_16bit);

        if (loss_4bit >= loss_1bit) {
            LOG_ERR("FAIL: 4-bit loss should be < 1-bit loss\n");
            return 1;
        }
        LOG_INF("PASS: Higher bitwidth gives lower sensitivity\n");

        LOG_INF("\n=== All Layers (bitwidth=1) ===\n");
        for (int i = 0; i < 32; i++) {
            float loss = test_params.get_quality_loss(i, 1);
            LOG_INF("Layer %2d: %f\n", i, loss);
        }

        LOG_INF("\n=== Test Complete ===\n");
        return 0;
    }

    // Adaptive quantization mode is required
    if (!params.ada_quant) {
        LOG_ERR("\nError: --ada-quant flag is required for this tool.\n");
        LOG_ERR("This is the offline scheduling tool for adaptive quantization.\n");
        LOG_ERR("Use --ada-quant to enable it.\n\n");
        print_usage(argc, argv);
        return 1;
    }

    // Load the model (for structure only - fp16 or base model)
    auto mparams = llama_model_params_from_gpt_params(params);
    struct llama_context_params cparams = llama_context_params_from_gpt_params(params);
    struct llama_model * model = nullptr;

    std::string model_path = params.model;
    if (model_path.empty()) {
        model_path = "/root/prima.cpp/models/Meta-Llama-3-8B-Instruct-fp16.gguf";
        LOG_INF("No model specified, using default: %s\n", model_path.c_str());
    }

    model = llama_load_model_from_file(model_path.c_str(), mparams);

    if (model == NULL) {
        LOG_ERR("Failed to load model '%s'\n", model_path.c_str());
        return -1;
    }
    llama_model_loader * ml = llama_model_load(model_path.c_str(), model, &mparams);

    // Initialize llama context for distributed setup
    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == nullptr) {
        LOG_ERR("Failed to create llama context\n");
        llama_free_model(model);
        return -1;
    }

    device_info dev_info;
    dev_info.rank = params.rank;
    uint32_t n_world   = params.n_world;
    uint32_t my_rank   = params.rank;
    dev_info.next_ip = params.next_node_ip.c_str();
    bool auto_schedule = params.n_layer_window[0] == 0;

    uint32_t n_layer_window[32] = {0}, n_gpu_layers[32] = {0};

    // initialize sockets
    llama_init_sockets(lctx, n_world, my_rank);

    // broadcast startup args
    struct startup_args args;
    if (my_rank == 0){
        args.should_profile = auto_schedule;
        args.n_ctx          = params.n_ctx;
    }

    llama_bcast_startup_args(lctx, my_rank, &args);

    if (my_rank > 0) {
        // receive startup args
        auto_schedule = args.should_profile;
        params.n_ctx  = args.n_ctx;
        cparams.n_ctx = args.n_ctx;
    }

    // if n_world > 1 and need auto schdule, then prifile
    if (auto_schedule){
        // get device profile
        LOG_INF("\nstart profiling this device, this may take some seconds ...\n");
        dev_info.rank = params.rank;
        dev_info.next_ip = params.next_node_ip.c_str();
        if (n_world > 1) {
            llama_profile_device(&dev_info, model, ml, params.gpu_mem, params.n_predict, params.n_ctx, params.cpuparams.n_threads, params.flash_attn, nullptr, params.ada_quant);
        }
    }
    // sychronize device profile to the master node
    NodeType node_type = NodeType::NODE_TYPE_WORKER;
    char is_forwarder[32] = {0};
    if (my_rank == 0) {
        if (auto_schedule) {
            std::vector<device_info> dev_info_set(n_world);
            dev_info_set[0] = dev_info;

            llama_gather_device_info(lctx, dev_info_set.data());
            device_print_props      (dev_info_set.data(), n_world, model, cparams, params.ada_quant);

            // assign layers to devices and remove weak devices
            // TODO: Improve ada-quant result logging and output formatting
            if (params.ada_quant) {
                LOG_INF("║       Starting Adaptive Quantization MILP Optimization         ║\n");
            }

            // Run layer allocation and exit with appropriate status
            if (assign_layers_and_select_devices(n_world, dev_info_set, n_layer_window, n_gpu_layers, model, cparams)) {
                LOG_INF("%s: Layer allocation completed successfully\n", __func__);

                // Notify other ranks before exiting
                llama_bcast_adaquant_exit(lctx);

                // Cleanup
                char *stop_signal = nullptr;
                llama_free_sockets(lctx, &stop_signal);
                llama_free(lctx);
                llama_free_model(model);
                exit(0);
            } else {
                LOG_ERR("%s: Layer allocation failed\n", __func__);
                llama_free(lctx);
                llama_free_model(model);
                exit(-1);
            }
        } else {
            // use the user-defined n_layer_window
            std::copy(std::begin(params.n_layer_window), std::end(params.n_layer_window), n_layer_window);
            llama_bcast_layer_setup(lctx, n_layer_window, nullptr);
        }
    } else {
        if (auto_schedule){
            llama_send_device_info (lctx, &dev_info);
            int recv_result = llama_recv_layer_setup (lctx, n_layer_window, n_gpu_layers);

            // Handle ada-quant exit signal
            if (recv_result == -2) {
                LOG_INF("%s: ada_quant mode - optimization complete on master, exiting\n", __func__);
                llama_free(lctx);
                llama_free_model(model);
                exit(0);
            }
        } else {
            llama_recv_layer_setup (lctx, n_layer_window, n_gpu_layers);
        }
    }

    // Should not reach here for workers
    llama_free(lctx);
    llama_free_model(model);
    return 0;
}
