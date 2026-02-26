#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "llama.h"
#include "profiler.h"
#include <fstream>
#include <sstream>
#include <cmath>

// Local function to convert ftype to bitwidth
static int ftype_to_bitwidth(int ftype) {
    // Strip the GUESSED flag (1024)
    int ftype_clean = ftype & ~1024;

    switch (ftype_clean) {
        case 0:  return 32; // LLAMA_FTYPE_ALL_F32
        case 1:  return 16; // LLAMA_FTYPE_MOSTLY_F16
        case 2:  return 4;  // LLAMA_FTYPE_MOSTLY_Q4_0
        case 3:  return 4;  // LLAMA_FTYPE_MOSTLY_Q4_1
        case 7:  return 8;  // LLAMA_FTYPE_MOSTLY_Q8_0
        case 8:  return 5;  // LLAMA_FTYPE_MOSTLY_Q5_0
        case 9:  return 5;  // LLAMA_FTYPE_MOSTLY_Q5_1
        case 10: return 2;  // LLAMA_FTYPE_MOSTLY_Q2_K
        case 11: return 3;  // LLAMA_FTYPE_MOSTLY_Q3_K_S
        case 12: return 3;  // LLAMA_FTYPE_MOSTLY_Q3_K_M
        case 13: return 3;  // LLAMA_FTYPE_MOSTLY_Q3_K_L
        case 14: return 4;  // LLAMA_FTYPE_MOSTLY_Q4_K_S
        case 15: return 4;  // LLAMA_FTYPE_MOSTLY_Q4_K_M
        case 16: return 5;  // LLAMA_FTYPE_MOSTLY_Q5_K_S
        case 17: return 5;  // LLAMA_FTYPE_MOSTLY_Q5_K_M
        case 18: return 6;  // LLAMA_FTYPE_MOSTLY_Q6_K
        case 19: return 2;  // LLAMA_FTYPE_MOSTLY_IQ2_XXS
        case 20: return 2;  // LLAMA_FTYPE_MOSTLY_IQ2_XS
        case 21: return 2;  // LLAMA_FTYPE_MOSTLY_Q2_K_S
        case 22: return 3;  // LLAMA_FTYPE_MOSTLY_IQ3_XS
        case 23: return 3;  // LLAMA_FTYPE_MOSTLY_IQ3_XXS
        case 24: return 1;  // LLAMA_FTYPE_MOSTLY_IQ1_S
        case 25: return 4;  // LLAMA_FTYPE_MOSTLY_IQ4_NL
        case 26: return 3;  // LLAMA_FTYPE_MOSTLY_IQ3_S
        case 27: return 3;  // LLAMA_FTYPE_MOSTLY_IQ3_M
        case 28: return 2;  // LLAMA_FTYPE_MOSTLY_IQ2_S
        case 29: return 2;  // LLAMA_FTYPE_MOSTLY_IQ2_M
        case 30: return 4;  // LLAMA_FTYPE_MOSTLY_IQ4_XS
        case 31: return 1;  // LLAMA_FTYPE_MOSTLY_IQ1_M
        case 32: return 16; // LLAMA_FTYPE_MOSTLY_BF16
        case 36: return 1;  // LLAMA_FTYPE_MOSTLY_TQ1_0
        case 37: return 2;  // LLAMA_FTYPE_MOSTLY_TQ2_0
        default: return 0;
    }
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

static void export_model_stats_to_csv(const device_info * dev_info, const struct llama_model * model, llama_model_loader * ml) {
    const char * csv_filename = "Model_Statistics.csv";

    // Check if file exists to determine if we need to write header
    bool needs_header = true;
    std::ifstream check_file(csv_filename);
    if (check_file.good()) {
        std::string first_line;
        std::getline(check_file, first_line);
        if (!first_line.empty() && first_line.find("Model,output_f32_f32") == 0) {
            needs_header = false;
        }
    }
    check_file.close();

    // Open file in append mode
    std::ofstream csv_file(csv_filename, std::ios::app);
    if (!csv_file.is_open()) {
        LOG_WRN("%s: failed to open CSV file: %s\n", __func__, csv_filename);
        return;
    }

    // Write header if needed
    if (needs_header) {
        csv_file << "Model,output_f32_f32,output_f16_f32,output_q2k_f32,output_q4k_f32,output_q5k_f32,output_q6k_f32,output_iq2xxs_f32,output_iq2xs_f32,output_q50_f32,output_q80_f32,output_iq1s_f32,output_iq4nl_f32,output_iq1m_f32,"
                 << "layer_f32_f32,layer_f16_f32,layer_q2k_f32,layer_q4k_f32,layer_q5k_f32,layer_q6k_f32,layer_iq2xxs_f32,layer_iq2xs_f32,layer_q50_f32,layer_q80_f32,layer_iq1s_f32,layer_iq4nl_f32,layer_iq1m_f32,"
                 << "input_f32,input_f16,input_q2k,input_q4k,input_q5k,input_q6k,input_iq2xxs,input_iq2xs,input_q50,input_q80,input_iq1s,input_iq4nl,input_iq1m,"
                 << "layer_f32,layer_f16,layer_q2k,layer_q4k,layer_q5k,layer_q6k,layer_iq2xxs,layer_iq2xs,layer_q50,layer_q80,layer_iq1s,layer_iq4nl,layer_iq1m,"
                 << "output_f32,output_f16,output_q2k,output_q4k,output_q5k,output_q6k,output_iq2xxs,output_iq2xs,output_q50,output_q80,output_iq1s,output_iq4nl,output_iq1m,"
                 << "nb_input,nb_layer,nb_output\n";
    }

    // Get architecture from GGUF metadata
    char arch_buf[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
    std::string arch(arch_buf);

    // Get model size from parameter count (convert to billions)
    uint64_t n_params = llama_model_n_params(model);

    // Get ftype from GGUF metadata "general.file_type"
    char ftype_buf[32] = {0};
    int ftype = 0;
    if (llama_model_meta_val_str(model, "general.file_type", ftype_buf, sizeof(ftype_buf)) > 0) {
        ftype = std::stoi(ftype_buf);
    }
    int bitwidth = ftype_to_bitwidth(ftype);

    // Format: architecture_sizeB_bitwidth (e.g., llama_8B_2)
    std::string model_id = arch + "_" + std::to_string(bitwidth);

    // Write data row
    csv_file << model_id << ","
             << dev_info->model_flops.output_f32_f32 << ","
             << dev_info->model_flops.output_f16_f32 << ","
             << dev_info->model_flops.output_q2k_f32 << ","
             << dev_info->model_flops.output_q4k_f32 << ","
             << dev_info->model_flops.output_q5k_f32 << ","
             << dev_info->model_flops.output_q6k_f32 << ","
             << dev_info->model_flops.output_iq2xxs_f32 << ","
             << dev_info->model_flops.output_iq2xs_f32 << ","
             << dev_info->model_flops.output_q50_f32 << ","
             << dev_info->model_flops.output_q80_f32 << ","
             << dev_info->model_flops.output_iq1s_f32 << ","
             << dev_info->model_flops.output_iq4nl_f32 << ","
             << dev_info->model_flops.output_iq1m_f32 << ","

             << dev_info->model_flops.layer_f32_f32 << ","
             << dev_info->model_flops.layer_f16_f32 << ","
             << dev_info->model_flops.layer_q2k_f32 << ","
             << dev_info->model_flops.layer_q4k_f32 << ","
             << dev_info->model_flops.layer_q5k_f32 << ","
             << dev_info->model_flops.layer_q6k_f32 << ","
             << dev_info->model_flops.layer_iq2xxs_f32 << ","
             << dev_info->model_flops.layer_iq2xs_f32 << ","
             << dev_info->model_flops.layer_q50_f32 << ","
             << dev_info->model_flops.layer_q80_f32 << ","
             << dev_info->model_flops.layer_iq1s_f32 << ","
             << dev_info->model_flops.layer_iq4nl_f32 << ","
             << dev_info->model_flops.layer_iq1m_f32 << ","

             << dev_info->model_params.input_f32 << ","
             << dev_info->model_params.input_f16 << ","
             << dev_info->model_params.input_q2k << ","
             << dev_info->model_params.input_q4k << ","
             << dev_info->model_params.input_q5k << ","
             << dev_info->model_params.input_q6k << ","
             << dev_info->model_params.input_iq2xxs << ","
             << dev_info->model_params.input_iq2xs << ","
             << dev_info->model_params.input_q50 << ","
             << dev_info->model_params.input_q80 << ","
             << dev_info->model_params.input_iq1s << ","
             << dev_info->model_params.input_iq4nl << ","
             << dev_info->model_params.input_iq1m << ","

             << dev_info->model_params.layer_f32 << ","
             << dev_info->model_params.layer_f16 << ","
             << dev_info->model_params.layer_q2k << ","
             << dev_info->model_params.layer_q4k << ","
             << dev_info->model_params.layer_q5k << ","
             << dev_info->model_params.layer_q6k << ","
             << dev_info->model_params.layer_iq2xxs << ","
             << dev_info->model_params.layer_iq2xs << ","
             << dev_info->model_params.layer_q50 << ","
             << dev_info->model_params.layer_q80 << ","
             << dev_info->model_params.layer_iq1s << ","
             << dev_info->model_params.layer_iq4nl << ","
             << dev_info->model_params.layer_iq1m << ","

             << dev_info->model_params.output_f32 << ","
             << dev_info->model_params.output_f16 << ","
             << dev_info->model_params.output_q2k << ","
             << dev_info->model_params.output_q4k << ","
             << dev_info->model_params.output_q5k << ","
             << dev_info->model_params.output_q6k << ","
             << dev_info->model_params.output_iq2xxs << ","
             << dev_info->model_params.output_iq2xs << ","
             << dev_info->model_params.output_q50 << ","
             << dev_info->model_params.output_q80 << ","
             << dev_info->model_params.output_iq1s << ","
             << dev_info->model_params.output_iq4nl << ","
             << dev_info->model_params.output_iq1m << ","

             << dev_info->model_bytes.nb_input << ","
             << dev_info->model_bytes.nb_layer << ","
             << dev_info->model_bytes.nb_output << "\n";

    csv_file.close();
    LOG_INF("%s: model statistics exported to %s\n", __func__, csv_filename);
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    // load the model and apply lora adapter, if any
    auto mparams = llama_model_params_from_gpt_params(params);
    struct llama_context_params cparams = llama_context_params_from_gpt_params(params);

    struct llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        return -1;
    }

    llama_model_loader * ml = llama_model_load(params.model.c_str(), model, &mparams);

    // Scan tensor min/max values
    llama_scan_tensor_minmax(ml);

    device_info dev_info;
    llama_profile_device(&dev_info, model, ml, params.gpu_mem, params.n_predict, params.n_ctx, params.cpuparams.n_threads, params.flash_attn, nullptr);
    device_print_props(&dev_info, 1, model, cparams, false);  // profile_tool doesn't use ada-quant
    // export_model_stats_to_csv(&dev_info, model, ml);

    llama_free_model(model);
    return 0;
}
