# bonsai-turing

High-throughput CUDA inference engine for **PrismML Ternary Bonsai 2.5 27B (`PQ2_0-mtp`)** on **NVIDIA Turing (`sm_75`, e.g. Quadro RTX 8000 / RTX 6000 / T4 / RTX 2080 Ti)**, built on top of `PrismML-Eng/llama.cpp` (`910442b`).

Several optimizations (lossless 2-nibble `mma.m8n8k32.s4` `PQ2_0` GEMM, fused 1-pass `FWHT(1024) → 2-nibble`, 3D `concat_dim0_transpose_u32`, batched multi-seq GDN `SET_ROWS` + `rpl_identity` in-place replay, split multi-seq MTP catch-up, adaptive KV `n_kv` padding, and GPU greedy argmax) also accelerate `PQ2_0` / Qwen3.5 / MTP workloads across `sm_75`–`sm_89`.

---

## Headline Benchmarks

**Hardware:** Single **NVIDIA Quadro RTX 8000 (48 GB GDDR6, 672 GB/s, 72 SMs, 576 Turing Tensor Cores, `sm_75`)**, SM clock locked at **1260 MHz** with a **200 W** power cap.  
**Model:** `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf` (27.32B params, 6.97 GiB, 2.13 bpw, 48 GDN + 16 Full-Attention layers + 1-layer MTP draft head).

| Workload | Stock `910442b` (`ub=512`) | Stock `910442b` (`ub=2048`) | **bonsai-turing (Lossless)** | **bonsai-turing (Opt-in INT4 `N>=256`)** | Speedup vs Stock |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Prefill `pp2048` (`f16` KV)** | `337.4 tok/s` | `791.5 tok/s` | `1,765+ tok/s` | **`3,008.5 tok/s`** | **8.9×** (`3.8×` vs `ub=2048`) |
| **Prefill `pp4096` (`f16` KV)** | — | — | — | **`2,934.8 tok/s`** | — |
| **Single-Stream `tg128`** | — | `42.2 tok/s` | **`55.7 tok/s`** | **`55.7 tok/s`** *(100% lossless)* | **1.32×** |
| **Single-Stream E2E MTP Decode (Code `n=7`)** | `41.2 tok/s` | — | **`123.9 tok/s`** | **`123.9 tok/s`** *(100% lossless)* | **3.01×** |
| **Single-Stream E2E MTP Decode (JSON `n=15`)** | — | — | **`311.6 tok/s`** | **`311.6 tok/s`** *(100% lossless)* | **~7.5×** |
| **Multi-Agent Steady-State (`-np 16 n=15`)** | — | `155.1 tok/s` *(`-np 8`)* | **`850.4 tok/s`** (`53.1/req`) | **`1,114.2 tok/s`** (`69.6/req`) | **5.5× – 7.2×** |
| **Multi-Agent Steady-State (`-np 24/32 n=15`)** | — | `155.1 tok/s` *(`-np 8`)* | **`922.8 tok/s`** *(`-np 32`, `28.8/req`)* | **`1,220.6 tok/s`** *(`-np 24`, `50.9/req`)* | **5.9× – 7.9×** |

### Quality & LoRA Parity (`b=512`, 12 WikiText chunks)

| Mode | PPL (`b=512`) | Δ PPL vs Stock | Logit Corr | Same Top-1 | Code Exact Match | Abliterate LoRA Parity |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Stock `910442b` (FP16/MMQ) | `4.0607 ± 0.1795` | baseline | `100.00%` | `100.00%` | `1089 / 1089` | baseline |
| **Lossless INT8 (`PQ2_NARROW_INT8` + `PQ2_CUBLAS_I8`)** | **`4.0584 ± 0.1794`** | **`-0.06%`** (`ub=64`: `+0.02%`) | **`99.99%`** | **`99.41%`** | **`1089 / 1089`** | **100% char-for-char** |
| **Opt-in Prefill/Verify INT4 (`N >= 256`, `clip=2.35×RMS`)** | `4.2152 ± 0.1882` | `+3.80%` | `99.1%` | `92.88%` | **`1089 / 1089`** | **100% char-for-char** |

> Note: Single-stream decode (`N <= 16`) and small-batch multi-agent decode (`N < 256`) **always** use the 100% lossless 2-nibble INT8 path even when `GGML_CUDA_PQ2_PREFILL_INT4=1` is enabled.

---

## Key Optimizations

1. **Lossless 2-Nibble INT8-via-INT4 Tensor Core GEMM (`pq2_narrow2d_i8_kernel` & `pq2_narrow2d_swiglu_kernel`):**
   - Decomposes exact 8-bit activations into high/low signed 4-bit nibbles (`q = 16*hi + lo`) and executes exact 8-bit × ternary matmuls on Turing's 2×-throughput `mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32` tensor cores (`128×64` & `64×64` 2D warp-tiled `uint4` loads, `4×4` L2 swizzle, 128 regs, 16 warps/SM) with zero 27 GB weight dequantization to HBM.
   - Fused 1-pass `FWHT(1024) → 2-nibble` activation packing + fused `SwiGLU` gate/up epilogue.
   - Handles `N = 5..512` (100% lossless), routing `lm_head` (`M > 65536`) at `N >= 256` to `PQ2_CUBLAS_I8`.
2. **Opt-in Prefill/Verify INT4 (`GGML_CUDA_PQ2_PREFILL_INT4=1`, `N >= 256`):**
   - 1-pass `FWHT(1024) → INT4` (`clip = 2.35 × RMS`) + Turing `mma.m8n8k32.s4` GEMM achieving **`3,008.5 tok/s`** (`pp2048`) and **`1,114–1,220 tok/s`** multi-agent decode.
3. **Turing `PQ2_0` Single-Token Decode (`mmvq`) + `GDN_MMA`:**
   - Tuned `mmvq` for `sm_75` (`55.7 tok/s` `tg128`) and chunked Tensor Core GDN prefill (`GGML_CUDA_GDN_MMA=1`, `GGML_CUDA_GDN_FAST_F16=1`).
4. **Multi-Agent GDN & Graph-Reuse Optimizations:**
   - **`rpl_identity` in-place GDN replay** (`src/models/qwen35.cpp`, `src/llama-graph.cpp`): Skips the 48-layer `GET_ROWS` state gather (`9.6 GB/round` at `-np 32`) when `rpl_row(i) == kv_head + i` for all sequences, falling back automatically on cell swaps.
   - **Batched multi-seq GDN `SET_ROWS` fusion** (`gated_delta_net.cu`): Fuses `n_seqs × 48` (`768–1,536`) `SET_ROWS` launches into `48` vectorized `float4` launches (`100%` cell-swap safe).
   - **3D `concat_dim0_transpose_u32`** (`concat.cu`): Generalizes fused transpose-concat across `ne[2]`.
   - **Split multi-seq MTP catch-up** (`common/speculative.cpp`): Separates variable-length catch-up rows from the fixed-shape `n_draft × K` chained draft pass.
   - **Adaptive KV `n_kv` padding** (`src/llama-kv-cache.cpp`): Pads `n_kv` to `4096`/`8192` for large contexts, lifting MTP draft graph reuse from `0/16` (`58 ms` rebuild/round) to **`16/16` (`0.09 ms`/round)**.
   - **Zero-copy GPU greedy argmax** (`ggml_backend_cuda_argmax_f32_rows_async` + `k=1` `common_sampler_fast_topk`): Eliminates 254–509 MB/round DtoH logits transfers (`0.04 ms` sampling).

---

## Build

```bash
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server llama-bench llama-perplexity -j $(nproc)
```

---

## Recommended `llama-server` Launch

### 100% Lossless Decode + 3,008 tok/s Prefill (Recommended for Single-User / `omp` / Coding)

```bash
GGML_CUDA_PQ2_NARROW_INT8=1 \
GGML_CUDA_PQ2_PREFILL_INT4=1 \
GGML_CUDA_PQ2_CUBLAS_I8=1 \
GGML_CUDA_PQ2_FUSE_ADD=2 \
GGML_CUDA_GDN_MMA=1 \
GGML_CUDA_GDN_FAST_F16=1 \
GGML_CUDA_PQ2_ALIGN_CACHE=1 \
GGML_CUDA_PQ2_TURING_TUNE=1 \
LLAMA_MTP_CHAIN=1 \
LLAMA_MTP_DRAFT_VOCAB=16384 \
./build/bin/llama-server \
  -m /path/to/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 \
  -c 65536 -np 4 -b 4096 -ub 2048 --spec-draft-n-ubatch 1024 \
  --fit off --jinja --min-p 0 --no-ui \
  --spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max 15 --spec-draft-p-min 0.0
```

- For **100% lossless multi-agent decode (`922.8 tok/s` at `-np 32 n=15`)**, set `GGML_CUDA_PQ2_PREFILL_INT4=0`.
- For **1,114–1,220 tok/s multi-agent decode (`-np 16..24 n=15`)**, keep `GGML_CUDA_PQ2_PREFILL_INT4=1`.

---

## License

MIT License (upstream `llama.cpp` / `PrismML-Eng/llama.cpp`).
