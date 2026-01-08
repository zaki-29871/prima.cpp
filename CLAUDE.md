# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

prima.cpp is a distributed implementation of llama.cpp that enables running large language models (70B+ parameters) on low-resource home clusters. It implements ring-topology distributed inference with memory-mapped model loading, heterogeneity-aware workload distribution, and speculative decoding.

## Build Commands

**Standard Build:**
```bash
make -j$(nproc)
```

**With CUDA Support:**
```bash
make GGML_CUDA=1 -j$(nproc)
```

**With HiGHS (required for rank 0 device):**
```bash
make USE_HIGHS=1 -j$(nproc)
```

**Combined (rank 0 with GPU):**
```bash
make GGML_CUDA=1 USE_HIGHS=1 -j$(nproc)
```

**Debug Mode:**
```bash
make LLAMA_DEBUG=1 -j$(nproc)
```

**macOS without Metal (for large models):**
```bash
make LLAMA_NO_METAL=1 -j$(nproc)
```

## Main Executables

- `llama-cli`: Command-line inference client
- `llama-server`: HTTP server for API endpoints
- `llama-speculative`: Speculative decoding inference
- `profile-tool`: Device profiling utility
- `llama-gguf-split`: Model splitting utility

## Test Commands

**Run all tests:**
```bash
make test
```

**Run specific test categories:**
```bash
# Run tokenizer tests
./tests/test-tokenizer-0
./tests/test-tokenizer-1-bpe
./tests/test-tokenizer-1-spm

# Run grammar tests
./tests/test-llama-grammar
./tests/test-grammar-parser
./tests/test-grammar-integration

# Run sampling tests
./tests/test-sampling

# Run quantization tests
./tests/test-quantize-fns
./tests/test-quantize-perf
```

## Architecture Overview

### Core Components

**Distributed Inference System:**
- Ring-topology communication using ZeroMQ (`network-utils.cpp/h`)
- Heterogeneity-aware workload scheduler using HiGHS optimization
- Memory-mapped model loading with OS-managed page cache
- Prefetching mechanism for overlapping computation and I/O

**Model Support:**
- GGUF format quantized models (Q4K, Q6K, Q80, IQ1)
- Llama, Qwen/QwQ, and DeepSeek model families
- Dynamic batching for concurrent requests

**Key Files:**
- `src/llama.cpp`: Core LLM inference implementation
- `src/network-utils.cpp/h`: Cross-device communication layer  
- `common/profiler.cpp/h`: Device capability profiling
- `common/speculative.cpp/h`: Speculative decoding implementation
- `examples/server/server.cpp`: HTTP API server

### Distributed Setup

**Single Device (degrades to llama.cpp):**
```bash
./llama-cli -m model.gguf -c 1024 -p "prompt" -n 256 -ngl 30
```

**Multi-Device Ring Topology:**
```bash
# Rank 0 (head device)
./llama-cli -m model.gguf --world 4 --rank 0 --master IP0 --next IP1 --prefetch

# Rank 1 (worker)
./llama-cli -m model.gguf --world 4 --rank 1 --master IP0 --next IP2 --prefetch --gpu-mem 8

# Rank 2 (worker)
./llama-cli -m model.gguf --world 4 --rank 2 --master IP0 --next IP3 --prefetch --gpu-mem 11

# Rank 3 (worker)
./llama-cli -m model.gguf --world 4 --rank 3 --master IP0 --next IP0 --prefetch
```

### Server Mode

**Launch distributed server:**
```bash
# Rank 0 (API server)
./llama-server -m model.gguf --world 2 --rank 0 --master IP0 --next IP1 --prefetch --host 127.0.0.1 --port 8080 -np 4 --cont-batching

# Rank 1 (worker)
./llama-cli -m model.gguf --world 2 --rank 1 --master IP0 --next IP0 --prefetch
```

**API Usage:**
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 200, "temperature": 0.7, "stream": true}'
```

## Python Environment Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Model conversion scripts:**
```bash
python convert_hf_to_gguf.py --help
python convert_lora_to_gguf.py --help
```

## Development Notes

- The system automatically profiles devices and assigns optimal workloads unless manual layer distribution is specified with `-lw`
- Use `--prefetch` for overlapping disk I/O with computation
- Add `--force` for aggressive prefetching (test before using in production)
- Communication ports default to 9000 (data) and 10000 (signals), configurable with `--data-port` and `--signal-port`
- Model files should be stored on SSD for optimal performance

## Manual Workload Distribution

```bash
# Manual layer assignment (16 layers per device)
./llama-cli -m model.gguf --world 4 --rank 0 --prefetch -lw "16,16,16,16"

# Manual GPU layers (8 out of 16 layers on GPU)
./llama-cli -m model.gguf --world 4 --rank 1 --prefetch -ngl 8
```

## Device Profiling

```bash
./profile-tool -m model.gguf
```