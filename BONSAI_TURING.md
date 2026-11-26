# bonsai-turing

High-throughput CUDA inference engine for **PrismML Ternary Bonsai 2.5 27B (`PQ2_0-mtp`)** on **NVIDIA Turing (`sm_75`, e.g. Quadro RTX 8000 / RTX 6000 / T4 / RTX 2080 Ti)**, built on top of `PrismML-Eng/llama.cpp` (`910442b`).

Several optimizations (lossless 2-nibble `mma.m8n8k32.s4` `PQ2_0` GEMM with wave-balanced split-K, fused 1-pass `FWHT(1024) → 2-nibble` / `B_i8`, coalesced SMEM `float4` sliding-window `cpy_f32_batched_conv3_fast`, `float4` ILP GDN replay, batched multi-seq GDN `SET_ROWS` + `rpl_identity` in-place replay, split multi-seq MTP catch-up, adaptive KV `n_kv` padding, and GPU greedy argmax) also accelerate `PQ2_0` / Qwen3.5 / MTP workloads across `sm_75`–`sm_89`.

---

## Headline Benchmarks

**Hardware:** Single **NVIDIA Quadro RTX 8000 (48 GB GDDR6, 672 GB/s, 72 SMs, 576 Turing Tensor Cores, `sm_75`)**, SM clock locked at **1260 MHz** with a **200 W** power cap.  
**Model:** `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf` (27.32B params, 6.97 GiB, 2.13 bpw, 48 GDN + 16 Full-Attention layers + 1-layer MTP draft head).

| Workload | Stock `910442b` (`ub=512`) | Stock `910442b` (`ub=2048`) | **bonsai-turing (100% Lossless)** | **bonsai-turing (Opt-in INT4 `N>=256`)** | Speedup vs Stock |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Prefill `pp2048` (`f16` KV)** | `337.4 tok/s` | `791.5 tok/s` | `1,765+ tok/s` | **`3,008.5 tok/s`** | **8.9×** (`3.8×` vs `ub=2048`) |
| **Prefill `pp4096` (`f16` KV)** | — | — | — | **`2,934.8 tok/s`** | — |
| **Single-Stream `tg128`** | — | `42.2 tok/s` | **`55.76 tok/s`** | **`55.76 tok/s`** *(100% lossless)* | **1.32×** |
| **Single-Stream E2E MTP Decode (Code `n=7`)** | `41.2 tok/s` | — | **`123.9 tok/s`** | **`123.9 tok/s`** *(100% lossless)* | **3.01×** |
| **Single-Stream E2E MTP Decode (JSON `n=15`)** | — | — | **`302.1–311.6 tok/s`** | **`302.1–311.6 tok/s`** *(100% lossless)* | **~7.5×** |
| **Multi-Agent Steady-State (`-np 16 n=15`, `f16` KV)** | — | `155.1 tok/s` *(`-np 8`)* | **`927.7 tok/s`** (`58.0/req`) | **`1,275.4 tok/s`** (`79.7/req`, `917.3` full E2E) | **6.0× – 8.2×** |
| **Multi-Agent Steady-State (`-np 24 n=15`)** | — | `155.1 tok/s` *(`-np 8`)* | **`987.6 tok/s`** *(`f16` KV, `41.2/req`)* | **`1,387.0 tok/s`** *(`q8_0` KV, `57.8/req`)* | **6.4× – 8.9×** |
| **Multi-Agent Steady-State (`-np 32 n=15`)** | — | `155.1 tok/s` *(`-np 8`)* | **`1,001.0 tok/s`** *(`f16` KV, `31.3/req`)* | **`1,330.4–1,354.4 tok/s`** *(`q8_0` KV, `41.6/req`)* | **6.5× – 8.7×** |

### Quality, KL-Divergence & LoRA Parity (`b=512`, 12 chunks = 6,144 tokens)

| Mode | `ub` / KV | PPL (`b=512`) | Δ PPL vs Stock | Logit Corr | Same Top-1 | Valid JSON (`-np 16..32`) | Abliterate LoRA Parity |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| Stock `910442b` (FP16/MMQ) | `ub=512`, `q8_0` | `4.0607 ± 0.1796` | baseline | `100.00%` | `100.00%` | — | baseline |
| **100% Lossless (`PQ2_NARROW_INT8` + `PQ2_CUBLAS_I8`)** | `ub=256`, `f16` | **`4.0597 ± 0.1795`** | **`-0.02%`** | — | — | **`24/24` – `32/32`** | **100% (`1446/1446` chars)** |
| **100% Lossless (`PQ2_NARROW_INT8` + `PQ2_CUBLAS_I8`)** | `ub=256`, `q8_0` | **`4.0602 ± 0.1795`** | **`-0.01%`** | **`99.99%`** | **`99.44%`** | **`32/32`** | **100% (`1446/1446` chars)** |
| **100% Lossless (`PQ2_NARROW_INT8` + `PQ2_CUBLAS_I8`)** | `ub=512`, `q8_0` | **`4.0608 ± 0.1796`** | **`+0.002%`** | **`99.99%`** | **`99.58%`** | **`32/32`** | **100% (`1446/1446` chars)** |
| **Opt-in Prefill/Verify INT4 (`N >= 256`, `clip=2.35×RMS`)** | `ub=256`, `q8_0` (`f16`) | `4.2187` (`4.2151`) | `+3.89%` (`+3.80%`) | `99.1%` | `92.16%` | **`16/16` – `32/32`** | **100% (`1446/1446` chars)** |
| **Opt-in Prefill/Verify INT4 (`N >= 256`, `clip=2.35×RMS`)** | `ub=512`, `q8_0` | `4.2152 ± 0.1882` | `+3.80%` | `99.1%` | `92.88%` | **`32/32`** | **100% (`1446/1446` chars)** |

> Note: Single-stream decode (`N <= 16`) and small-batch multi-agent decode (`N < 256`) **always** use the 100% lossless 2-nibble INT8 path even when `GGML_CUDA_PQ2_PREFILL_INT4=1` is enabled.

---

## Key Optimizations

1. **Lossless 2-Nibble INT8-via-INT4 Tensor Core GEMM (`pq2_narrow2d_i8_kernel` & `pq2_narrow2d_swiglu_kernel`):**
   - Decomposes exact 8-bit activations into high/low signed 4-bit nibbles (`q = 16*hi + lo`) and executes exact 8-bit × ternary matmuls on Turing's 2×-throughput `mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32` tensor cores (`128×64` & `64×64` 2D warp-tiled `uint4` loads, `4×4` L2 swizzle, 128 regs, 16 warps/SM) with zero 27 GB weight dequantization to HBM.
   - **Wave-balanced split-K (`2`/`3`)** on `N = 129..512, M <= 12288` eliminates tail-wave SM underutilization on 72 SMs (`pp256`: `1,384.8`, `pp384`: `1,472.6`, `pp512`: `1,502.6 tok/s`).
   - Fused 1-pass `FWHT(1024) → 2-nibble` activation packing, fused `SwiGLU` gate/up epilogue, and `MUL_MAT + ADD` residual fusion (`N > 32`).
   - Handles `N = 5..512` (100% lossless), routing `lm_head` (`M > 65536`) at `N >= 256` directly into `PQ2_CUBLAS_I8` with fused `FWHT(1024) → B_i8` packing.
2. **Opt-in Prefill/Verify INT4 (`GGML_CUDA_PQ2_PREFILL_INT4=1`, `N >= 256`):**
   - 1-pass `FWHT(1024) → INT4` (`clip = 2.35 × RMS`) + Turing `mma.m8n8k32.s4` GEMM achieving **`3,008.5 tok/s`** (`pp2048`) and **`1,275–1,387 tok/s`** multi-agent decode.
   - OOM-safe VRAM headroom guards with automatic pool-buffer fallback, `M >= M` prefix cache reuse (eliminating `output.weight`/`head_sub` cache thrashing), and full pre-capture warm-up for 100% capture-safe CUDA graphs up to `-np 32`.
3. **Turing `PQ2_0` Single-Token Decode (`mmvq`) + `GDN_MMA`:**
   - Tuned `mmvq` for `sm_75` (`55.76 tok/s` `tg128`) and chunked Tensor Core GDN prefill (`GGML_CUDA_GDN_MMA=1`, `GGML_CUDA_GDN_FAST_F16=1`).
4. **Multi-Agent GDN, Conv3 & Graph-Reuse Optimizations:**
   - **Coalesced SMEM `float4` sliding-window `cpy_f32_batched_conv3_fast`** (`cpy.cu`): Eliminates 16× redundant strided conv3 state reads across multi-agent verify batches (`+87 tok/s` at `-np 32`).
   - **`float4` ILP `gdn_replay_seq_cuda` + `rpl_identity` in-place GDN replay** (`gated_delta_net.cu`, `src/models/qwen35.cpp`, `src/llama-graph.cpp`): Vectorizes recurrent GDN state replay and skips the 48-layer `GET_ROWS` state gather (`9.6 GB/round` at `-np 32`) when `rpl_row(i) == kv_head + i`, falling back automatically on cell swaps.
   - **Batched multi-seq GDN `SET_ROWS` fusion** (`gated_delta_net.cu`, `src/models/delta-net-base.cpp`): Fuses `n_seqs × 48` (`768–1,536`) `SET_ROWS` launches into `48` vectorized `float4` launches (`100%` cell-swap safe via per-seq `wrow_dev` device pointers).
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

cmake --build build --target llama-server llama-bench llama-perplexity llama-completion -j $(nproc)
```

---

## Recommended `llama-server` Launch

### 100% Lossless Decode (`N < 256`) + 3,008 tok/s Prefill + 1,275–1,387 tok/s Multi-Agent Verify

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

- For **100% lossless multi-agent decode (`1,001.0 tok/s` at `-np 32 n=15`, `f16` KV)**, set `GGML_CUDA_PQ2_PREFILL_INT4=0`.
- For **1,275–1,387 tok/s multi-agent decode (`-np 16..24 n=15`)**, keep `GGML_CUDA_PQ2_PREFILL_INT4=1` (use `-ctk q8_0 -ctv q8_0` for `-np 24..32` to stay comfortably within 48 GB VRAM).

---

## License

MIT License (upstream `llama.cpp` / `PrismML-Eng/llama.cpp`).
