#include "profiler.h"
#include <cstdio>
#include <fstream>
#include <iomanip>

// Save sensitivity cache to CSV file
// Format: layer_idx,bitwidth,sensitivity
bool save_sensitivity_cache_to_csv(
    const bitwidth_model_params& params,
    const char* output_path,
    int num_layers = 32,
    const std::vector<int>& bitwidths = {1, 2, 4, 8, 16}
) {
    std::ofstream csv_file(output_path);
    if (!csv_file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open output file: %s\n", output_path);
        return false;
    }

    // Write header
    csv_file << "layer_idx,bitwidth,sensitivity\n";

    // Write data
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int bw : bitwidths) {
            float sensitivity = params.get_quality_loss(layer, bw);
            csv_file << layer << "," << bw << "," << std::scientific << std::setprecision(6) << sensitivity << "\n";
        }
    }

    csv_file.close();
    printf("Saved sensitivity cache to: %s\n", output_path);
    printf("  - Layers: %d\n", num_layers);
    printf("  - Bitwidths: ");
    for (size_t i = 0; i < bitwidths.size(); ++i) {
        printf("%d%s", bitwidths[i], (i < bitwidths.size() - 1) ? ", " : "\n");
    }
    printf("  - Total entries: %d\n", num_layers * (int)bitwidths.size());

    return true;
}

int main(int argc, char** argv) {
    const char* variance_csv = (argc > 1) ? argv[1] : "/root/prima.cpp/tensor_capture_stats.csv";
    const char* minmax_csv = (argc > 2) ? argv[2] : "/root/prima.cpp/tensor_minmax_stats_all.csv";

    printf("=== Layer Sensitivity Test ===\n");
    printf("Variance CSV: %s\n", variance_csv);
    printf("MinMax CSV: %s\n\n", minmax_csv);

    bitwidth_model_params params;

    // 加载数据
    if (!params.load_sensitivity_data(variance_csv, minmax_csv)) {
        fprintf(stderr, "ERROR: Failed to load sensitivity data\n");
        return 1;
    }

    printf("Data loaded successfully\n");
    printf("Has sensitivity data: %s\n\n", params.has_sensitivity_data() ? "yes" : "no");

    // 测试层 0 的不同位宽敏感度
    printf("=== Layer 0 Sensitivity at Different Bitwidths ===\n");
    printf("Bitwidth | Sensitivity    | Scale (relative to 1-bit)\n");
    printf("---------|----------------|-------------------------\n");

    float baseline_loss = 0.0f;
    for (int bw : {1, 2, 4, 8, 16}) {
        float loss = params.get_quality_loss(0, bw);
        if (bw == 1) baseline_loss = loss;
        float scale = (baseline_loss > 0) ? (loss / baseline_loss) : 0.0f;
        printf("%8d | %14.6f | %6.4f%%\n", bw, loss, scale * 100.0f);
    }

    // 验证：位宽越高，敏感度越低
    printf("\n=== Validation ===\n");
    float loss_1bit = params.get_quality_loss(0, 1);
    float loss_4bit = params.get_quality_loss(0, 4);
    float loss_16bit = params.get_quality_loss(0, 16);

    printf("Layer 0, bitwidth 1:  %f\n", loss_1bit);
    printf("Layer 0, bitwidth 4:  %f\n", loss_4bit);
    printf("Layer 0, bitwidth 16: %f\n", loss_16bit);

    if (loss_4bit >= loss_1bit) {
        printf("ERROR: 4-bit loss (%f) should be < 1-bit loss (%f)\n", loss_4bit, loss_1bit);
        return 1;
    }
    if (loss_16bit >= loss_4bit) {
        printf("ERROR: 16-bit loss (%f) should be < 4-bit loss (%f)\n", loss_16bit, loss_4bit);
        return 1;
    }

    printf("VALIDATED: Higher bitwidth gives lower sensitivity\n");

    // 打印所有层的敏感度（位宽 1）
    printf("\n=== All Layers Sensitivity (bitwidth=1) ===\n");
    for (int i = 0; i < 32; i++) {
        float loss = params.get_quality_loss(i, 1);
        printf("Layer %2d: %f\n", i, loss);
    }

    printf("\n=== Test Complete ===\n");

    // Save sensitivity cache to CSV file
    const char* output_csv = (argc > 3) ? argv[3] : "/root/prima.cpp/sensitivity_cache.csv";
    printf("\n=== Saving Sensitivity Cache ===\n");
    if (!save_sensitivity_cache_to_csv(params, output_csv, 32, {1, 2, 4, 8, 16})) {
        fprintf(stderr, "ERROR: Failed to save sensitivity cache\n");
        return 1;
    }

    printf("\nCache saved successfully!\n");
    printf("You can now load this cache in your code using:\n");
    printf("  bitwidth_model_params params;\n");
    printf("  params.load_sensitivity_cache_from_csv(\"%s\");\n", output_csv);

    return 0;
}
