#ifndef PROFILER_H
#define PROFILER_H

#include "ggml.h"
#include "llama.h"

#define EPS                  1e-9f
#define DISK_TEST_TOTAL_BYTE 500L * 1024 * 1024
#define DISK_TEST_SEQ_BLOCK  100L * 1024 * 1024
#define DISK_TEST_RND_BLOCK  4096
#define MEM_TEST_BLOCK_SIZE  64 * 1024


struct cpu_props {
    const char * name;
    const char * description;
    uint32_t     cores;
    float        flops_f32_f32;     // in GFLOPS
    float        flops_f16_f32;     // in GFLOPS
    float        flops_q2k_f32;     // in GFLOPS
    float        flops_q4k_f32;     // in GFLOPS
    float        flops_q5k_f32;     // in GFLOPS
    float        flops_q6k_f32;     // in GFLOPS
    float        flops_iq2xxs_f32;  // in GFLOPS
    float        flops_iq2xs_f32;   // in GFLOPS
    float        flops_q50_f32;     // in GFLOPS    
    float        flops_q80_f32;     // in GFLOPS
    float        flops_iq1s_f32;    // in GFLOPS
    float        flops_iq4nl_f32;   // in GFLOPS
    float        flops_iq1m_f32;    // in GFLOPS

    cpu_props()
        : name            (""), 
          description     (""), 
          cores           (0), 
          flops_f32_f32   (0.0f), 
          flops_f16_f32   (0.0f), 
          flops_q2k_f32   (0.0f),
          flops_q4k_f32   (0.0f),
          flops_q5k_f32   (0.0f),
          flops_q6k_f32   (0.0f),
          flops_iq2xxs_f32(0.0f),
          flops_iq2xs_f32 (0.0f),
          flops_q50_f32   (0.0f),
          flops_q80_f32   (0.0f),
          flops_iq1s_f32  (0.0f), 
          flops_iq4nl_f32 (0.0f),
          flops_iq1m_f32  (0.0f)
    {}
};

struct memory_info {
    float        total_physical;     // in GiB
    float        available_physical; // in GiB
    float        used_can_swap;      // in GiB
    float        total_swap;         // in GiB
    float        available_swap;     // in GiB
    float        cpu_read_ram_bw;    // in GB/s
    float        mem_cpy_delay;      // in ms

    memory_info() : 
        total_physical    (0.0f), 
        available_physical(0.0f), 
        used_can_swap     (0.0f),
        total_swap        (0.0f), 
        available_swap    (0.0f), 
        cpu_read_ram_bw   (0.0f),
        mem_cpy_delay     (0.0f) {}
};

struct gpu_support {
    bool         metal;
    bool         cuda;
    bool         vulkan;
    bool         kompute;
    bool         gpublas;
    bool         blas;
    bool         sycl;

    gpu_support() : 
        metal  (false), 
        cuda   (false), 
        vulkan (false), 
        kompute(false), 
        gpublas(false), 
        blas   (false), 
        sycl   (false) {}
};

struct gpu_props {
    const char * name;
    const char * description;
    float        memory_free;               // in GiB
    float        memory_total;              // in GiB
    float        metal_read_vram_bw;        // in GB/s
    float        metal_flops_f32_f32;       // in GFLOPS
    float        metal_flops_f16_f32;       // in GFLOPS
    float        metal_flops_q2k_f32;       // in GFLOPS
    float        metal_flops_q4k_f32;     // in GFLOPS
    float        metal_flops_q5k_f32;     // in GFLOPS
    float        metal_flops_q6k_f32;     // in GFLOPS
    float        metal_flops_iq2xxs_f32;  // in GFLOPS
    float        metal_flops_iq2xs_f32;   // in GFLOPS
    float        metal_flops_q50_f32;     // in GFLOPS
    float        metal_flops_q80_f32;     // in GFLOPS
    float        metal_flops_iq1s_f32;    // in GFLOPS
    float        metal_flops_iq4nl_f32;   // in GFLOPS
    float        metal_flops_iq1m_f32;    // in GFLOPS
    float        metal_mem_cpy_delay;     // in ms
    float        cuda_read_vram_bw;       // in GB/s
    float        cuda_flops_f32_f32;      // in GFLOPS
    float        cuda_flops_f16_f32;      // in GFLOPS
    float        cuda_flops_q2k_f32;      // in GFLOPS
    float        cuda_flops_q4k_f32;      // in GFLOPS
    float        cuda_flops_q5k_f32;      // in GFLOPS
    float        cuda_flops_q6k_f32;      // in GFLOPS
    float        cuda_flops_iq2xxs_f32;   // in GFLOPS
    float        cuda_flops_iq2xs_f32;    // in GFLOPS
    float        cuda_flops_q50_f32;      // in GFLOPS
    float        cuda_flops_q80_f32;      // in GFLOPS
    float        cuda_flops_iq1s_f32;     // in GFLOPS
    float        cuda_flops_iq4nl_f32;    // in GFLOPS
    float        cuda_flops_iq1m_f32;     // in GFLOPS
    float        cuda_mem_cpy_delay;      // in ms
    float        cuda_pcie_bw;            // in GB/s (measured CPU<->GPU transfer bandwidth)

    gpu_props() : 
        name                    (""), 
        description             (""), 
        memory_free             (0.0f), 
        memory_total            (0.0f), 
        metal_read_vram_bw      (0.0f),
        metal_flops_f32_f32     (0.0f), 
        metal_flops_f16_f32     (0.0f),
        metal_flops_q2k_f32     (0.0f),
        metal_flops_q4k_f32     (0.0f),
        metal_flops_q5k_f32     (0.0f),
        metal_flops_q6k_f32     (0.0f),
        metal_flops_iq2xxs_f32  (0.0f),
        metal_flops_iq2xs_f32   (0.0f),
        metal_flops_q50_f32     (0.0f),
        metal_flops_q80_f32     (0.0f),
        metal_flops_iq1s_f32    (0.0f),
        metal_flops_iq4nl_f32   (0.0f),
        metal_flops_iq1m_f32    (0.0f),
        metal_mem_cpy_delay     (0.0f),
        cuda_read_vram_bw       (0.0f),
        cuda_flops_f32_f32      (0.0f), 
        cuda_flops_f16_f32      (0.0f), 
        cuda_flops_q2k_f32      (0.0f),
        cuda_flops_q4k_f32      (0.0f),
        cuda_flops_q5k_f32      (0.0f),
        cuda_flops_q6k_f32      (0.0f),
        cuda_flops_iq2xxs_f32   (0.0f),
        cuda_flops_iq2xs_f32    (0.0f),
        cuda_flops_q50_f32      (0.0f),
        cuda_flops_q80_f32      (0.0f),
        cuda_flops_iq1s_f32     (0.0f),
        cuda_flops_iq4nl_f32    (0.0f),
        cuda_flops_iq1m_f32     (0.0f),
        cuda_mem_cpy_delay      (0.0f),
        cuda_pcie_bw            (0.0f) {}
};

struct model_flops {
    float   inp_embd_ms;
    int64_t output_f32_f32;
    int64_t output_f16_f32;
    int64_t output_q2k_f32;
    int64_t output_q4k_f32;
    int64_t output_q5k_f32;
    int64_t output_q6k_f32;
    int64_t output_iq2xxs_f32;
    int64_t output_iq2xs_f32;
    int64_t output_q50_f32;
    int64_t output_q80_f32;
    int64_t output_iq1s_f32;
    int64_t output_iq4nl_f32;
    int64_t output_iq1m_f32;
    int64_t layer_f32_f32;
    int64_t layer_f16_f32;
    int64_t layer_q2k_f32;
    int64_t layer_q4k_f32;
    int64_t layer_q5k_f32;
    int64_t layer_q6k_f32;
    int64_t layer_iq2xxs_f32;
    int64_t layer_iq2xs_f32;
    int64_t layer_q50_f32;
    int64_t layer_q80_f32;
    int64_t layer_iq1s_f32;
    int64_t layer_iq4nl_f32;
    int64_t layer_iq1m_f32;

    model_flops() : 
        inp_embd_ms        (0.0f),
        output_f32_f32     (0), 
        output_f16_f32     (0),
        output_q2k_f32     (0),
        output_q4k_f32     (0),
        output_q5k_f32     (0),
        output_q6k_f32     (0), 
        output_iq2xxs_f32  (0),
        output_iq2xs_f32   (0),
        output_q50_f32     (0),
        output_q80_f32     (0),
        output_iq1s_f32    (0),
        output_iq4nl_f32   (0),
        output_iq1m_f32    (0),
        layer_f32_f32      (0),
        layer_f16_f32      (0),
        layer_q2k_f32      (0),
        layer_q4k_f32      (0),
        layer_q5k_f32      (0),
        layer_q6k_f32      (0),
        layer_iq2xxs_f32   (0),
        layer_iq2xs_f32    (0),
        layer_q50_f32      (0),
        layer_q80_f32      (0),
        layer_iq1s_f32     (0),
        layer_iq4nl_f32    (0), 
        layer_iq1m_f32     (0)
        {}
};

struct model_params {
    int64_t input_f32;
    int64_t input_f16;
    int64_t input_q2k;
    int64_t input_q4k;
    int64_t input_q5k;
    int64_t input_q6k;
    int64_t input_iq2xxs;
    int64_t input_iq2xs;
    int64_t input_q50;
    int64_t input_q80;
    int64_t input_iq1s;
    int64_t input_iq4nl;
    int64_t input_iq1m;
    int64_t output_f32;
    int64_t output_f16;
    int64_t output_q2k;
    int64_t output_q4k;
    int64_t output_q5k;
    int64_t output_q6k;
    int64_t output_iq2xxs;
    int64_t output_iq2xs;
    int64_t output_q50;
    int64_t output_q80;
    int64_t output_iq1s;
    int64_t output_iq4nl;
    int64_t output_iq1m;
    int64_t layer_f32;
    int64_t layer_f16;
    int64_t layer_q2k;
    int64_t layer_q4k;
    int64_t layer_q5k;
    int64_t layer_q6k;
    int64_t layer_iq2xxs;
    int64_t layer_iq2xs;
    int64_t layer_q50;
    int64_t layer_q80;
    int64_t layer_iq1s;
    int64_t layer_iq4nl;
    int64_t layer_iq1m;

    model_params() :
        input_f32       (0),
        input_f16       (0),
        input_q2k       (0),
        input_q4k       (0),
        input_q5k       (0),
        input_q6k       (0),
        input_iq2xxs    (0),
        input_iq2xs     (0),
        input_q50       (0),
        input_q80       (0),
        input_iq1s      (0),
        input_iq4nl     (0),
        input_iq1m      (0),
        output_f32      (0),
        output_f16      (0),
        output_q2k      (0),
        output_q4k      (0),
        output_q5k      (0),
        output_q6k      (0),
        output_iq2xxs   (0),
        output_iq2xs    (0),
        output_q50      (0),
        output_q80      (0),
        output_iq1s     (0),
        output_iq4nl    (0),
        output_iq1m     (0),
        layer_f32       (0),
        layer_f16       (0),
        layer_q2k       (0),
        layer_q4k       (0),
        layer_q5k       (0),
        layer_q6k       (0),
        layer_iq2xxs    (0),
        layer_iq2xs     (0),
        layer_q50       (0),
        layer_q80       (0),
        layer_iq1s      (0),
        layer_iq4nl     (0), 
        layer_iq1m      (0)
        {}
};

struct model_bytes {
    int64_t nb_input;
    int64_t nb_layer;
    int64_t nb_output;

    // used to estimate the compute buffer size 
    int64_t nb_output_w;
    int64_t nb_output_norm_w;
    int64_t nb_attn_norm_w;
    int64_t nb_attn_q_w;
    int64_t nb_ffn_gate_w;
    int64_t nb_ffn_down_w;

    model_bytes() :
        nb_input        (0),
        nb_layer        (0),
        nb_output       (0), 
        nb_output_w     (0),
        nb_output_norm_w(0),
        nb_attn_norm_w  (0),
        nb_attn_q_w     (0),
        nb_ffn_gate_w   (0),
        nb_ffn_down_w   (0) {}
};

struct disk_props {
    float read_seq_bw;  // in GB/s
    float read_rnd_bw;  // in GB/s
    float write_seq_bw; // in GB/s
    float write_rnd_bw; // in GB/s

    disk_props() :
        read_seq_bw (0.0f),
        read_rnd_bw (0.0f),
        write_seq_bw(0.0f),
        write_rnd_bw(0.0f) {}
};

struct startup_args{
    bool     should_profile;
    uint32_t n_ctx;
};

// Forward declaration for model_statistics_entry
struct model_statistics_entry;

struct device_info {
    uint32_t            rank;
    const char *        device_name;
    const char *        device_os;
    const char *        next_ip;
    struct disk_props   disk;
    struct cpu_props    cpu_props;
    struct memory_info  memory;
    struct gpu_support  gpu_support;
    struct gpu_props    gpu_props;
    struct model_flops  model_flops;
    struct model_params model_params;
    struct model_bytes  model_bytes;

    std::vector<model_statistics_entry> model_stats_all;

    device_info() : 
        rank(0), 
        device_name(""), 
        device_os(""),
        next_ip(""),
        disk(),
        cpu_props(), 
        memory(), 
        gpu_support(), 
        gpu_props(), 
        model_flops(),
        model_params(),
        model_bytes() {}
};

struct TopoRebuildHelperInfo{
    struct device_info dev_info;
    char               is_forwarder;
    
    TopoRebuildHelperInfo():
        dev_info(),
        is_forwarder(0){}
    
    void   deserialize(const char * buffer);
    size_t serialize(char ** buffer) const;
};

enum profiler_backend_type {
    PROFILER_BACKEND_TYPE_CPU   = 0,
    PROFILER_BACKEND_TYPE_METAL = 1,
    PROFILER_BACKEND_TYPE_CUDA  = 2,
};

enum profiler_layer_type {
    PROFILER_LAYER_INPUT   = 0,
    PROFILER_LAYER_OUTPUT  = 1,
    PROFILER_LAYER_BACKEND = 2,
};

const char * device_name(void); 
const char * device_os(void);

uint32_t device_cpu_cores         (void);
float    device_cpu_flops         (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, int n_threads);
float    device_metal_flops       (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
float    device_cuda_flops        (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
float    device_inp_embd_delay    (struct llama_model * model, enum ggml_type src0t, int n_tokens, int n_threads);
uint64_t device_physical_memory   (bool available);
uint64_t device_swap_memory       (bool available);
uint64_t device_swappable_memory  ();
void     device_disk_seq_bw       (float * read_seq_bw, float * write_seq_bw, int n_threads);
void     device_disk_rnd_bw       (float * read_rnd_bw, float * write_rnd_bw, int n_threads);
float    device_memory_bw         (int n_thread);
float    device_cpu_mem_copy      (struct llama_model * model, int n_threads);
float    device_metal_mem_copy    (struct llama_model * model);
float    device_cuda_mem_copy     (struct llama_model * model);
float    device_metal_read_vram_bw();
float    device_cuda_read_vram_bw ();
float    device_cuda_pcie_bw      (); // Measure CPU<->GPU PCIe bandwidth (Linux only)
void     device_get_props         (struct llama_model * model, int device, struct ggml_backend_dev_props * props); 
void     device_print_props       (struct device_info * dev_info_set, int n, struct llama_model * model, const struct llama_context_params cparams, bool ada_quant = false);
void     device_export_csv        (const struct device_info * dev_info_set, int n, struct llama_model * model, const char * output_path);

uint64_t device_host_physical_memory(bool available);

int      device_has_metal  (void);
int      device_has_cuda   (void);
int      device_has_vulkan (void);
int      device_has_kompute(void);
int      device_has_gpublas(void);
int      device_has_blas   (void);
int      device_has_sycl   (void);

size_t   serialize  (const struct device_info * dev_info, char ** buffer);
size_t   deserialize(const char * buffer, struct device_info * dev_info);
struct model_statistics_entry {
    std::string model_id;       // e.g., "llama_1", "llama_2", "llama_4", "llama_8", "llama_16"
    struct model_flops flops;   // FLOPS for each quantization type
    struct model_params params;  // Parameter counts for each quantization type
    struct model_bytes bytes;    // Byte sizes for input/layer/output

    model_statistics_entry() : model_id(""), flops(), params(), bytes() {}
};

// Tensor activation variance statistics captured during inference
// Uses Welford's online algorithm for O(1) memory variance computation
struct tensor_variance_entry {
    std::string tensor_name;   // e.g., "blk.0.attn_q.in"
    int32_t     samples;       // Number of samples captured
    float       variance;      // Variance of the tensor activation

    tensor_variance_entry() : tensor_name(""), samples(0), variance(0.0f) {}
};

// Weight min-max statistics for quantization sensitivity calculation
// CSV scale column contains pre-computed scaling factor S_W(b) for each bitwidth
struct weight_minmax_entry {
    std::string weight_name;    // e.g., "blk.0.attn_q.weight"
    int32_t     bit_width;      // 1, 2, 4, 8, 16
    int64_t     elements;       // Number of parameters D_W
    float       min_val;        // Minimum weight value
    float       max_val;        // Maximum weight value
    float       range;          // max - min
    float       scale;          // CSV pre-computed scaling factor S_W(bit_width)

    weight_minmax_entry()
        : weight_name(""), bit_width(0), elements(0),
          min_val(0.0f), max_val(0.0f), range(0.0f), scale(0.0f) {}

    // Get scaling factor - directly use CSV scale value
    float get_scaling_factor() const {
        return scale;
    }

    // When falling back to other bitwidth, recompute scaling factor from range
    float get_scaling_factor_from_range(int target_bitwidth) const {
        if (target_bitwidth <= 0 || target_bitwidth > 16) return 0.0f;
        float denominator = static_cast<float>((1LL << target_bitwidth) - 1);
        return (denominator > 0.0f) ? (range / denominator) : 0.0f;
    }
};

// Load model statistics from CSV file
// Returns true on success, false on failure
bool load_model_statistics_from_csv(
    const char * csv_path,
    std::vector<model_statistics_entry>& entries
);

// Load tensor variance statistics from tensor_capture_stats.csv
// Format: layer_name,samples,variance
bool load_tensor_variance_from_csv(
    const char* csv_path,
    std::unordered_map<std::string, tensor_variance_entry>& variance_map
);

// Load weight minmax statistics from tensor_minmax_stats_all.csv
// Format: tensor_name,bit_width,elements,min,max,range,scale,type
bool load_weight_minmax_from_csv(
    const char* csv_path,
    std::unordered_map<int, std::unordered_map<std::string,
        std::unordered_map<int, weight_minmax_entry>>>& minmax_map,
    int32_t& n_layers
);

// Internal FLOPS measurement function with n_embd override support
// Allows profiling without loading a model (pass model=nullptr and n_embd_override=4096)
float device_flops_internal(
    struct llama_model * model,
    enum ggml_type src0t,
    enum ggml_type src1t,
    enum profiler_backend_type btype,
    int n_threads,
    int n_embd_override);

// ============================================================
// Adaptive Bitwidth Allocation Structures
// ============================================================

// Mapping from nominal bitwidth to ggml_type and model metadata
// LaTeX: b in B -> (ggml_type, model_suffix, quality_loss)
struct bitwidth_entry {
    int nominal_bitwidth;       // 1, 2, 4, 8, 16
    enum ggml_type weight_type; // GGML_TYPE_IQ1_S, GGML_TYPE_Q2_K, etc.
    const char* model_suffix;   // "_Q4_K_M", "_Q8_0", "_F16", etc.

    // Per-parameter statistics (computed from model metadata)
    float quality_loss;         // Delta Q for this bitwidth (Fisher-weighted outlier)

    // Actual model statistics loaded from CSV
    // Pointer to the full model statistics entry for this bitwidth
    // MUST NOT be nullptr - CSV data is required
    const model_statistics_entry* model_stats;

    bitwidth_entry()
        : nominal_bitwidth(0), weight_type(GGML_TYPE_F32), model_suffix(""),
          quality_loss(0.0f), model_stats(nullptr) {}

    bitwidth_entry(int bw, enum ggml_type wt, const char* suffix,
                   float qloss, const model_statistics_entry* stats)
        : nominal_bitwidth(bw), weight_type(wt), model_suffix(suffix),
          quality_loss(qloss), model_stats(stats) {}
};

// Container for bitwidth lookup table
// Provides mapping: bitwidth -> (layer_bytes, layer_flops, quality_loss)
struct bitwidth_model_params {
    std::unordered_map<int, bitwidth_entry> entries;

    // Default constructor - creates empty entries
    // Must use the CSV constructor to populate with actual model statistics
    bitwidth_model_params() {
        // Entries remain empty - CSV data is required for proper operation
    }

    // Constructor with model statistics from CSV
    // Parses model_stats_all to extract bitwidth-specific data
    bitwidth_model_params(const std::vector<model_statistics_entry>& model_stats_all) {
        // Helper function to extract bitwidth from model_id (e.g., "llama_1" -> 1)
        auto extract_bitwidth = [](const std::string& model_id) -> int {
            size_t underscore_pos = model_id.find('_');
            if (underscore_pos != std::string::npos) {
                std::string bw_str = model_id.substr(underscore_pos + 1);
                try {
                    return std::stoi(bw_str);
                } catch (...) {
                    return 0;
                }
            }
            return 0;
        };

        // Build bitwidth to model statistics mapping
        std::unordered_map<int, const model_statistics_entry*> bw_to_stats;
        for (const auto& entry : model_stats_all) {
            int bw = extract_bitwidth(entry.model_id);
            if (bw > 0) {
                bw_to_stats[bw] = &entry;
            }
        }

        // Initialize entries with actual CSV data
        // Map bitwidth to ggml_type and use CSV statistics
        entries[1]  = bitwidth_entry(1,  GGML_TYPE_IQ1_S,   "_IQ1_S",   2.0f,
                                    bw_to_stats.count(1)  ? bw_to_stats[1]  : nullptr);
        entries[2]  = bitwidth_entry(2,  GGML_TYPE_Q2_K,     "_Q2_K",     1.0f,
                                    bw_to_stats.count(2)  ? bw_to_stats[2]  : nullptr);
        entries[4]  = bitwidth_entry(4,  GGML_TYPE_Q4_K,     "_Q4_K_M",   0.3f,
                                    bw_to_stats.count(4)  ? bw_to_stats[4]  : nullptr);
        entries[8]  = bitwidth_entry(8,  GGML_TYPE_Q8_0,     "_Q8_0",     0.1f,
                                    bw_to_stats.count(8)  ? bw_to_stats[8]  : nullptr);
        entries[16] = bitwidth_entry(16, GGML_TYPE_F16,      "_F16",      0.0f,
                                    bw_to_stats.count(16) ? bw_to_stats[16] : nullptr);
    }

    const bitwidth_entry* get_entry(int bitwidth) const {
        auto it = entries.find(bitwidth);
        return (it != entries.end()) ? &(it->second) : nullptr;
    }

    enum ggml_type get_weight_type(int bitwidth) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        return e ? e->weight_type : GGML_TYPE_F32;
    }

    // LaTeX: beta_{i,b} - per-layer bytes at given bitwidth
    // CSV data is required - will assert if not available
    int64_t get_layer_bytes(int layer_idx, int bitwidth, const struct model_bytes& base) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        GGML_ASSERT(e != nullptr && "Bitwidth entry not found - CSV data required");
        GGML_ASSERT(e->model_stats != nullptr && "Model statistics (CSV data) required - cannot proceed without it");
        return e->model_stats->bytes.nb_layer;
    }

    // LaTeX: f_{i,b} - per-layer FLOPs at given bitwidth
    // Returns the complete model_flops structure containing FLOPs for all operation types
    // This allows the caller to compute execution time by matching each operation type
    // with the corresponding device performance (e.g., layer_q4k_f32 / cpu.flops_q4k_f32)
    const struct model_flops& get_layer_flops(int layer_idx, int bitwidth, const struct model_flops& base) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        // If CSV data is available, return the complete FLOPs structure
        // containing all operation types (F32, F16, Q2K, Q4K, etc.)
        if (e && e->model_stats != nullptr) {
            return e->model_stats->flops;
        }
        // Fallback to base FLOPs structure
        return base;
    }

    // LaTeX: Delta Q_{i,b} - quality loss at given bitwidth
    // Uses calculated layer sensitivity or falls back to hardcoded values
    float get_quality_loss(int layer_idx, int bitwidth) const;

    // Load sensitivity data from CSV files
    // Call this before using get_quality_loss() with layer_idx for accurate results
    bool load_sensitivity_data(const char* variance_csv_path, const char* minmax_csv_path);

    // Load pre-computed sensitivity cache from CSV file
    // CSV format: layer_idx,bitwidth,sensitivity
    bool load_sensitivity_cache_from_csv(const char* cache_csv_path);

    // Check if sensitivity data has been loaded
    bool has_sensitivity_data() const { return sensitivity_data_loaded; }

    // Clear loaded sensitivity data and reset to hardcoded mode
    void clear_sensitivity_data();

private:
    // Tensor variance statistics: tensor_name -> variance_entry
    std::unordered_map<std::string, tensor_variance_entry> tensor_variance_map;

    // Weight minmax statistics: layer_idx -> (weight_name -> (bitwidth -> minmax_entry))
    std::unordered_map<int, std::unordered_map<std::string,
        std::unordered_map<int, weight_minmax_entry>>> weight_minmax_map;

    // Sensitivity calculation cache: cache[layer_idx][bitwidth] = sensitivity_value
    mutable std::unordered_map<int, std::unordered_map<int, float>> sensitivity_cache;

    // Flag indicating if sensitivity data has been loaded
    bool sensitivity_data_loaded = false;

    // Number of layers in the model
    int32_t n_layers = 0;

    // Get activation variance for a specific input tensor
    float get_input_variance(int layer_idx, const std::string& input_name) const;

    // Get weight elements count and scaling factor
    // Returns: pair<elements_count, scaling_factor>
    std::pair<int64_t, float> get_weight_minmax_scale(
        int layer_idx, const std::string& weight_name, int bitwidth) const;

    // Calculate layer sensitivity for a given bitwidth
    // Formula: omega_{i,b} = sum_{o in O_i} [ D_Wo * (S_Wo(b))^2 * G(Xo) ]
    float calculate_layer_sensitivity(int layer_idx, int bitwidth) const;

public:
    // Check if bitwidth is supported
    bool has_bitwidth(int bitwidth) const {
        return entries.find(bitwidth) != entries.end();
    }

    // Check if bitwidth has actual CSV data
    bool has_csv_data(int bitwidth) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        return e && e->model_stats != nullptr;
    }

    // LaTeX: B_i^b - input layer bytes at bitwidth b
    // CSV data is required - will assert if not available
    int64_t get_input_bytes(int bitwidth) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        GGML_ASSERT(e != nullptr && "Bitwidth entry not found - CSV data required");
        GGML_ASSERT(e->model_stats != nullptr && "Model statistics (CSV data) required - cannot proceed without it");
        return e->model_stats->bytes.nb_input;
    }

    // LaTeX: B_o^b - output layer bytes at bitwidth b
    // CSV data is required - will assert if not available
    int64_t get_output_bytes(int bitwidth) const {
        const bitwidth_entry* e = get_entry(bitwidth);
        GGML_ASSERT(e != nullptr && "Bitwidth entry not found - CSV data required");
        GGML_ASSERT(e->model_stats != nullptr && "Model statistics (CSV data) required - cannot proceed without it");
        return e->model_stats->bytes.nb_output;
    }
};

// 2D allocation result: [device][bitwidth_index] -> layer count
// Output structure for assign_layers_to_device_adaptive
struct bitwidth_allocation {
    // LaTeX: collapsed from z_{i,m,b}^{tau} solution
    std::vector<std::vector<uint32_t>> n_layer_window;  // [m][b_idx] -> total layers on device m with bitwidth b
    std::vector<std::vector<uint32_t>> n_gpu_layers;    // [m][b_idx] -> GPU layers on device m with bitwidth b
    std::vector<int> selected_bitwidth_cpu;   // [m] -> optimal bitwidth for CPU backend
    std::vector<int> selected_bitwidth_gpu;   // [m] -> optimal bitwidth for GPU backend

    bitwidth_allocation() {}

    bitwidth_allocation(uint32_t n_world, size_t n_bitwidths)
        : n_layer_window(n_world, std::vector<uint32_t>(n_bitwidths, 0)),
          n_gpu_layers(n_world, std::vector<uint32_t>(n_bitwidths, 0)),
          selected_bitwidth_cpu(n_world, 0),
          selected_bitwidth_gpu(n_world, 0) {}

    void resize(uint32_t n_world, size_t n_bitwidths) {
        n_layer_window.assign(n_world, std::vector<uint32_t>(n_bitwidths, 0));
        n_gpu_layers.assign(n_world, std::vector<uint32_t>(n_bitwidths, 0));
        selected_bitwidth_cpu.assign(n_world, 0);
        selected_bitwidth_gpu.assign(n_world, 0);
    }

    // Get total layers assigned to device m (sum over all bitwidths)
    uint32_t total_layers(uint32_t m) const {
        uint32_t sum = 0;
        for (const auto& count : n_layer_window[m]) {
            sum += count;
        }
        return sum;
    }

    // Get total GPU layers assigned to device m
    uint32_t total_gpu_layers(uint32_t m) const {
        uint32_t sum = 0;
        for (const auto& count : n_gpu_layers[m]) {
            sum += count;
        }
        return sum;
    }
};

#endif // PROFILER_H
