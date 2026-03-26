# Study Plan: Blackwell GDN Prefill Implementation

**Branch under study:** `dianzhangchen:gdn_dev`
**Goal:** Deep understanding of the Blackwell GDN chunked prefill kernel, culminating in a proof-of-learning patch.

---

## Overview

5 commits, ~6400 lines added across 8 files. Adds a Blackwell (SM100/SM110) backend for the chunked Gated Delta Rule (GDN) linear attention prefill kernel, written entirely in CuTe-DSL (Python-level CUTLASS). The existing Hopper (SM90) path is preserved; a new dispatch layer routes to the right backend based on GPU architecture.

### Files Changed

| File | Lines | Role |
|------|-------|------|
| `flashinfer/gdn_kernels/blackwell_prefill/gdn.py` | +4686 | Main kernel (CuTe-DSL) |
| `flashinfer/gdn_kernels/blackwell_prefill/gdn_helpers.py` | +177 | SMEM layout helpers for tcgen05 |
| `flashinfer/gdn_kernels/blackwell_prefill/gdn_tile_scheduler.py` | +165 | Persistent/non-persistent tile scheduler |
| `flashinfer/gdn_kernels/blackwell_prefill/__init__.py` | +6 | Package init |
| `flashinfer/gdn_prefill.py` | +143/-48 | Dispatch layer (SM90 vs SM100/110) |
| `tests/gdn/test_gdn_prefill_blackwell.py` | +681 | Tests with reference impl |
| `benchmarks/bench_blackwell_gdn_prefill.py` | +503 | Benchmarks vs FLA baseline |

---

## Phase 1: Understand the Algorithm (no code yet)

**Goal:** Build intuition for *what* this kernel computes before looking at *how*.

### 1.1 Gated Delta Rule (GDN) Recap

Study the recurrent reference in `test_gdn_prefill_blackwell.py:44-72` (`recurrent_gated_delta_rule_ref`).

Key equations per timestep `t`:

```
h_t = h_{t-1} * exp(g_t) + k_t^T * beta_t * (v_t - h_{t-1} @ k_t)
o_t = q_t @ h_t
```

Where:
- `h` is the recurrent state (d x d matrix per head)
- `g` is the forget/decay gate (log-space, applied via exp)
- `beta` is the update gate (controls how much new info enters state)
- The "delta" is `(v_t - h_{t-1} @ k_t)` -- the *residual* between the actual value and what the state already predicts

**Exercise 1.1:** Write out the math for 3 timesteps with d=2. What is the role of `g` (gate/forget) vs `beta` (update gate)? Why is it called "delta" rule?

```
Answer:
```

**Exercise 1.2:** Compare to standard linear attention (`h_t = h_{t-1} + k_t^T v_t`, `o_t = q_t h_t`). What problem does the delta correction solve? (hint: think about what happens when the same key appears twice)

```
Answer:
```

### 1.2 Chunked Formulation

The kernel doesn't run token-by-token. It processes chunks of 128 tokens. Within a chunk:
- **Intra-chunk:** compute attention-like scores Q@K^T, apply gating, solve a linear system (matrix inversion) to get the "corrected" intra-chunk output
- **Inter-chunk:** propagate state `S` across chunks: `S_new = decay * S_old + update`

**Exercise 1.3:** Why chunk? What's the complexity of token-by-token recurrence vs chunked? (hint: sequential d^2 ops vs parallel GEMM)

```
Answer:
```

**Exercise 1.4:** The matrix inversion `(I - M)^{-1}` appears because the intra-chunk delta corrections are coupled -- token i's correction depends on token j < i within the same chunk. Why does this form a triangular system? Why is inversion feasible at chunk_size=128?

```
Answer:
```

### 1.3 GVA (Grouped Value Attention)

`h_v` can be a multiple of `h_q` (the reverse of GQA). `h_r = h_v / h_q` is the replication factor.

**Exercise 1.5:** In standard GQA, multiple query heads share one KV head. GVA flips this -- why would you want more value heads than query heads? What does this buy you in terms of state capacity?

```
Answer:
```

---

## Phase 2: Architecture & Dispatch Layer

**File:** `flashinfer/gdn_prefill.py`

### 2.1 Dispatch Refactor

The old `chunk_gated_delta_rule` is renamed to `chunk_gated_delta_rule_hopper`. A new top-level function dispatches:

```python
if is_sm90a_supported():    -> chunk_gated_delta_rule_hopper(...)
elif is_sm100a_supported(): -> chunk_gated_delta_rule_blackwell(...)
else:                       -> NotImplementedError
```

Key decorators: `@supported_compute_capability([90, 100, 110])` and `@backend_requirement({})`.

### 2.2 Blackwell Constraints vs Hopper

| Feature | Hopper (SM90) | Blackwell (SM100/110) |
|---------|--------------|----------------------|
| `g`, `beta` = None | Defaults to all-ones | **Not allowed** (raises error) |
| head_dim | 128 | 128 |
| Input dtypes | f16/bf16 | f16/bf16 |
| State dtype | f32 | f32 |
| Implementation | CUDA/Triton | CuTe-DSL |

**Exercise 2.1:** Why might Blackwell not support `g=None, beta=None`? (hint: the Hopper path handles this in the kernel; the Blackwell CuTe-DSL path expects the caller to provide them)

```
Answer:
```

---

## Phase 3: The Kernel Skeleton

**File:** `flashinfer/gdn_kernels/blackwell_prefill/gdn.py`

### 3.1 GDN.__init__ Configuration (lines 76-182)

Key parameters:
- **Tile sizes:** `chunk_size=128`, `head_dim=128`, all MMA tilers are 128x128xK
- **8 warps with specialized roles:**

```
Warp 0-3: cudacore warps  -- main computation, matrix inversion, gate cumsum
Warp 4:   MMA warp        -- tcgen05 tensor core UMMA instructions
Warp 5:   load warp       -- TMA async loads (Q, K, V, state)
Warp 6:   epilogue warp   -- TMA async stores (O, state output)
Warp 7:   gate/beta warp  -- loads g, beta into shared memory
```

- **TMEM layout** -- 512 columns of Tensor Memory, carefully partitioned:
  - Each MMA operation gets a non-overlapping TMEM region for its accumulator
  - e.g., `tmem_kkt_output=0`, `tmem_qkt_output=384`, `tmem_state=128`

- **Pipeline stages** for overlapping loads/computes/stores

**Exercise 3.1:** Draw the producer/consumer relationships between warps. For each pipeline (load_qk, load_v, load_state, mma_qk, mma_cudacore, gb_w0, epi), which warp produces and which consumes?

```
Warp    | Produces            | Consumes
--------|---------------------|-------------------
0-3     |                     |
4       |                     |
5       |                     |
6       |                     |
7       |                     |
```

### 3.2 GDN.kernel (lines 266-~1510) -- Warp-Specialized Entry Point

The `@cute.kernel` entry point:
1. TMA descriptor prefetch (load warp)
2. SharedStorage allocation via `SmemAllocator`
3. 7+ pipelines created (TMA loads, UMMA MMA, async copies)
4. Tile scheduler loop: iterate over chunks
5. `if warp_idx in cudacore_warp_ids:` -> main_loop
6. `elif warp_idx == mma_warp_id:` -> exec_mma
7. `elif warp_idx == load_warp_id:` -> TMA load loops
8. `elif warp_idx == epilogue_warp_id:` -> TMA store loops
9. `elif warp_idx == gb_warp_id:` -> gate/beta loads

**Key insight:** This is a *warp-specialized* kernel. Unlike typical kernels where all threads do the same thing, different warps run completely different code paths concurrently, communicating through shared memory and barrier pipelines.

---

## Phase 4: Core Compute Methods (the hard part)

### 4.1 main_loop (line 1511)

Per-chunk compute executed by cudacore warps (0-3). Sequence per chunk:

1. Wait for gate/beta from smem -> `chunk_local_cumsum` (cumulative gate sum)
2. `compute_gamma_tmem` -> decay factors for causal masking
3. Wait for QK^T MMA result -> apply gating/masking
4. **Matrix inversion** (2-level block inversion via `store_ivt_*` / `load_ivt_*`)
5. Apply beta, compute corrected values
6. Trigger state update MMA and output MMA
7. Write results to smem for epilogue

**Exercise 4.1:** Map each step in `main_loop` to the mathematical equation:

```
Step -> Equation
chunk_local_cumsum          -> ?
compute_gamma_tmem          -> ?
QK^T                        -> ?
Matrix inversion (I-M)^{-1} -> ?
State update S_new          -> ?
Output o = q @ S            -> ?
```

### 4.2 Matrix Inversion (lines 2620-3800)

The most complex part. Solves `(I - M)^{-1}` where M is lower-triangular (causal mask times gating). Uses **two-level recursive block inversion** with TFloat32 MMAs:

```
Level 0 (L0): 64x64 sub-blocks, using invert_sub_tiled_mma_ss_l0/ts_l0
Level 1 (L1): combines L0 results into 128x128, using ss_l1/ts_l1
```

The block inversion identity:
```
[A  0]^{-1}   [A^{-1}           0    ]
[C  D]       = [-D^{-1} C A^{-1}  D^{-1}]
```

**Exercise 4.2:** Why is this matrix lower-triangular? (hint: causal attention -- token i only attends to j <= i)

```
Answer:
```

**Exercise 4.3:** What is the complexity of this 2-level block inversion vs naive O(n^3)? Why use TFloat32 for the inversion but f16/bf16 for the main GEMMs?

```
Answer:
```

### 4.3 exec_mma (line 1740)

The MMA warp (warp 4) runs all tcgen05 UMMA instructions:
- QK^T, KK^T, KS^T
- Matrix inversion sub-GEMMs (L0 and L1)
- Output GEMMs (o_intra, q@S)
- State update GEMM

Uses TMEM for accumulator storage -- a Blackwell-specific feature where MMA results live in dedicated on-chip memory (not registers).

### 4.4 State Management

- `init_state_zeros` / `init_state_from_smem` -- handle first chunk
- `load_state_apply_gate` -- apply exponential decay to state between chunks
- `store_state_to_smem` / `store_o_smem` -- write to smem for epilogue TMA store

---

## Phase 5: Support Infrastructure

### 5.1 gdn_helpers.py -- SMEM Layout Construction

Three helpers for building swizzled shared memory layouts for Blackwell tcgen05:
- `make_smem_layout_a_kind` / `make_smem_layout_b_kind` -- MMA operand A/B
- `make_smem_layout_epi_kind` -- epilogue (output) layouts

These use `cute.nvgpu.tcgen05.SmemLayoutAtomKind` and handle K-major vs M/N-major, staging, and swizzle patterns.

### 5.2 gdn_tile_scheduler.py -- Work Distribution

`GdnStaticTileScheduler` distributes chunks across SMs:
- **Persistent mode:** `grid = min(sm_count, total_blocks)` CTAs; each CTA loops over work items via linear index
- **Non-persistent mode:** one CTA per tile, standard grid launch

Work item = `(chunk_idx, batch_idx, head_idx)`. The scheduler linearizes this 3D space.

**Exercise 5.1:** Compare to the SM90 GRMIS scheduler (MinHeap, 132 CTAs). What's similar/different?

```
Answer:
```

---

## Phase 6: Entry Point & Compilation

### 6.1 GDN.__call__ (line 3802) -- Host-Side Setup

1. Builds CuTe tensor layouts from raw pointers (complex strides for GVA)
2. Creates TMA copy atoms and descriptors
3. Sets up all SMEM layouts (calls helpers from gdn_helpers.py)
4. Computes SharedStorage layout and grid dimensions
5. Launches the kernel via `cute.launch_kernel`

### 6.2 chunk_gated_delta_rule (line 4548) -- Python API

1. Allocates output tensors
2. `_get_problem_size` (cached by shape)
3. JIT compilation via `cute.compile[EnableTVMFFI]` on first call, then cached in `_get_compiled_gdn_prefill_kernel` dict
4. Runs compiled kernel with raw `data_ptr()` pointers

**Exercise 6.1:** Trace a full call from user's `flashinfer.gdn_prefill.chunk_gated_delta_rule(q,k,v,g,beta)` to kernel launch. How many layers of dispatch are there?

```
Layer 1: gdn_prefill.chunk_gated_delta_rule -> arch dispatch
Layer 2: ...
Layer 3: ...
Layer 4: ...
```

---

## Phase 7: Tests & Benchmarks

### 7.1 Tests (`test_gdn_prefill_blackwell.py`)

Parametrized across:
- Fixed-length and variable-length (cu_seqlens)
- With/without initial state and output state
- GVA configurations (h_v = 1x, 2x, 4x of h_q)
- Reference: token-by-token recurrent implementation
- Tolerances: atol=1e-3, rtol=1e-3 for f16; 5e-3/1e-3 for bf16

### 7.2 Benchmarks (`bench_blackwell_gdn_prefill.py`)

Compares against FLA baseline (`fla.ops.gated_delta_rule.chunk`). Supports sweep mode across batch sizes and sequence lengths.

---

## Recommended Reading Order

| Order | File | Time |
|-------|------|------|
| 1 | `test_gdn_prefill_blackwell.py` (reference impl only) | 20 min |
| 2 | `flashinfer/gdn_prefill.py` (diff only) | 15 min |
| 3 | `gdn_tile_scheduler.py` | 20 min |
| 4 | `gdn_helpers.py` | 20 min |
| 5 | `gdn.py` -- `__init__` + `can_implement` | 30 min |
| 6 | `gdn.py` -- `chunk_gated_delta_rule` (bottom of file) | 15 min |
| 7 | `gdn.py` -- `__call__` | 30 min |
| 8 | `gdn.py` -- `kernel` (warp specialization structure) | 45 min |
| 9 | `gdn.py` -- `main_loop` | 60 min |
| 10 | `gdn.py` -- matrix inversion methods | 60 min |

---

## Key Concepts Reference

| Concept | What It Is |
|---------|-----------|
| **CuTe-DSL** | Python bindings for CUTLASS CuTe -- tensor layouts, TMA, UMMA |
| **tcgen05** | Blackwell's 5th-gen tensor core (UMMA instructions, TMEM) |
| **TMEM** | Tensor Memory -- Blackwell-only 512-column scratchpad for MMA accumulators |
| **Warp specialization** | Different warps run different code paths concurrently |
| **TMA** | Tensor Memory Accelerator -- hardware async copy engine |
| **Pipeline primitives** | Producer/consumer barriers for overlapping load/compute/store |
| **UMMA** | Unified Matrix Multiply-Accumulate -- Blackwell's MMA instruction |
| **TFloat32** | Truncated float32 used for higher-precision intermediate computation |

---

## Proof-of-Learning Patch (Mergeable Options)

**End goal:** A patch that (a) proves deep understanding of the Blackwell GDN kernel and (b) is genuinely useful to the repo and will be approved upstream.

### Option A: Blackwell-Optimized GDN Decode with F32x2 Packed FMA (BEST FIT)
**Why it merges:** The existing decode kernels (`gdn_decode_bf16_state.py`) explicitly note
"Can be optimized with packed F32x2 FMA for SM100+ in future releases" (line 37).
Currently they use scalar FMA that works on SM90+, but Blackwell has native `fma_packed_f32x2`
which doubles throughput for the state update. No one has done this yet.

**What you'd do:**
- Add a Blackwell-specific decode kernel path using `cute.arch.fma_packed_f32x2()`
- The scalar FMA helpers (`mul_f32`, `fma_f32` at lines 131-160) become packed equivalents
- Add SM100/110 dispatch in `gdn_decode.py` (same pattern as the prefill dispatch you studied)
- Benchmark: should see ~1.5-2x speedup on decode state update
- Tests: existing decode tests + SM100 skip logic

**Proves understanding of:** warp-level compute patterns, state update math, Blackwell ISA features,
dispatch architecture. Directly analogous to the prefill change you're studying.

**Difficulty:** Medium. The decode kernel is ~2000 lines (vs 4700 for prefill), simpler structure
(no matrix inversion, no warp specialization), and you have a clear before/after to benchmark.

### Option B: Fix SMEM Bank Conflicts in Blackwell Prefill Matrix Inversion
**Why it merges:** There are 4 explicit `# todo: remove smem bank conflict` comments in the
matrix inversion code (lines 2740, 2757, 2845, 2865 of `gdn.py`). These are in the
`store_ivt_smem_l0_ss_b` and `store_ivt_smem_l1_ss_b` methods -- the hot path of the
2-level block inversion.

**What you'd do:**
- Profile the kernel with Nsight Compute to quantify bank conflict overhead
- Redesign the SMEM layout for the inversion sub-tiles to avoid conflicts
  (likely need swizzled addressing or padding, similar to `gdn_decode_mtp.py:1720`
  which uses `stride = K + 4` to avoid bank conflicts)
- Benchmark before/after on the full prefill kernel
- The fix touches a small, self-contained part of the kernel

**Proves understanding of:** SMEM bank conflict mechanics, the matrix inversion data flow,
TMEM-to-SMEM copy patterns, and profiling methodology.

**Difficulty:** Medium-Hard. Requires Nsight Compute profiling on a Blackwell GPU and
understanding the exact access patterns in the inversion loop.

### Option C: Add bf16/fp16 State Support for Blackwell Prefill
**Why it merges:** The kernel header explicitly states "State input and output are in f32
(fp16/bf16 not supported yet)" (line 31 of `gdn.py`). The decode kernel already has a
`bf16_state` variant that's widely used. Reducing state from f32 to bf16 halves state
memory bandwidth -- critical for long sequences with many heads.

**What you'd do:**
- Add a bf16 accumulation path for the state tensor in `__call__` and `kernel`
- The state TMA loads/stores need new copy atoms for bf16
- The state update MMA (`update_s_tiled_mma`) accumulates in f32 but reads/writes bf16 state
- Add `state_dtype` parameter to `chunk_gated_delta_rule` API
- Tests: verify numerical accuracy vs f32 state (will need looser tolerances)

**Proves understanding of:** state lifecycle (init -> gate -> update -> store), TMA descriptor
setup, mixed-precision accumulation, the full host-to-device data flow.

**Difficulty:** Hard. Touches many parts of the kernel (TMA descriptors, SMEM layouts,
accumulator handling) but is a clear feature gap.

### Option D: Add Blackwell GDN Prefill to AOT Compilation + Benchmark Suite
**Why it merges:** The new kernel has no entry in `flashinfer/aot.py` (needed for
pre-compiled packages) and the benchmark isn't integrated into the unified
`benchmarks/flashinfer_benchmark.py` framework. These are table-stakes for landing a new kernel.

**What you'd do:**
- Register the Blackwell GDN prefill in `flashinfer/aot.py` for pre-compiled packages
- Integrate `bench_blackwell_gdn_prefill.py` into the unified benchmark framework
- Add the kernel to `flashinfer/__init__.py` exports properly for Blackwell
- Ensure `@flashinfer_api` logging works on the new dispatch path
- Add proper `@register_fake_op` for `torch.compile` compatibility on the Blackwell path

**Proves understanding of:** the full FlashInfer infrastructure (JIT, AOT, benchmarking,
torch.compile integration), and the dispatch/compilation flow of the new kernel.

**Difficulty:** Medium-Low. More infrastructure than kernel work, but requires understanding
how the kernel fits into the broader system.

### Recommendation

**Start with Option A (F32x2 decode)** if you want the clearest "I understood the GDN math and
Blackwell ISA" signal with a high chance of merging. It's the most explicitly requested
optimization in the codebase.

**Do Option B (bank conflicts)** if you want to go deep on the prefill kernel specifically
and have Blackwell GPU access for profiling.

**Do Option D (AOT + benchmarks)** as a complementary patch alongside A or B -- it's
lower risk and fills a real gap.

---

## Progress Tracker

- [ ] Phase 1: Algorithm understanding (exercises completed)
- [ ] Phase 2: Dispatch layer understood
- [ ] Phase 3: Kernel skeleton mapped
- [ ] Phase 4: Core compute methods traced
- [ ] Phase 5: Support infrastructure reviewed
- [ ] Phase 6: End-to-end call path traced
- [ ] Phase 7: Tests and benchmarks reviewed
- [ ] Proof-of-learning patch written and tested
