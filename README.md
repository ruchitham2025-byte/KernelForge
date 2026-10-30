# KernelForge: Autonomous Hardware-in-the-Loop GPU Kernel Synthesis

> **Nebius x NVIDIA Global AI Hackathon Submission**
> **LLM Engine:** `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`)
> **Target Hardware:** NVIDIA Tesla T4 (Turing Architecture, SM75, 16 GB GDDR6)
> **Environment:** PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0
> **Interactive Telemetry:** [View Live Dashboard & Replay](https://ruchitham2025-byte.github.io/KernelForge/dashboard/)

---

## 1. Overview & System Architecture

KernelForge is an autonomous hardware-in-the-loop agent that prompts an open NVIDIA LLM (Nemotron-3 Super 120B) to generate, benchmark, and iteratively repair OpenAI Triton GPU kernels directly on physical accelerator hardware.

```text
 ┌──────────────────────────────────────────────┐
 │ Nebius Token Factory                         │
 │ (nvidia/nemotron-3-super-120b-a12b)          │
 └──────────────────────┬───────────────────────┘
                        │ Generates candidate Triton kernel
                        ▼
 ┌──────────────────────────────────────────────┐
 │ Subprocess Execution Harness (runner.py)     │
 │ Target: NVIDIA Tesla T4 (Turing SM75)        │
 └───────────┬──────────────────────┬───────────┘
             │                      │
  Numerical drift / traceback   Pass + latency
             ▼                      ▼
 ┌─────────────────────────┐  ┌───────────────────────────┐
 │ Feedback injection into │  │ Update champion kernel &  │
 │ LLM context (max error, │  │ autopsy (interleaved      │
 │ stack tail, shape hints)│  │ timing vs baselines)      │
 └─────────────────────────┘  └───────────────────────────┘
```

The loop runs autonomously without human tuning: errors and execution timings are classified and fed back to the model context to drive iterative self-repair.

---

## 2. Benchmark Results & Baseline Integrity

All performance numbers are the **median of 7 interleaved rounds** using `triton.testing.do_bench(warmup=50, rep=200, return_mode="median")`, following a mandatory 1.0-second active GPU warm-up to stabilize clock frequencies and reduce thermal throttling artifacts.

- **Target Workload:** SwiGLU Epilogue (`out = (x @ W1.T + b1) * silu(x @ W2.T + b2)`)
- **Tensor Shapes:** `x (4096, 1024)`, `W1, W2 (2048, 1024)`, `b1, b2 (2048,)` in `torch.float16`
- **Numerical Tolerance:** `|y_cand - y_ref| <= 5e-3 + 1e-2 * |y_ref|` (`atol=5e-3, rtol=1e-2`), applied uniformly across all harnesses and `results_epilogue.json`
- **Verification Artifact:** Recorded in `results_epilogue.json` on a Tesla T4 running PyTorch `2.11.0+cu130` and Triton `3.6.0`

| Implementation | Architecture / Execution Path | Median Latency | Relative Speedup | Max Abs Error | Numerical Pass |
|:---|:---|:---|:---|:---|:---|
| **PyTorch Eager** | 2x cuBLAS GEMM + separate eager epilogue passes | 1.770 ms | 1.000x (baseline) | 0.0000 | Reference |
| **KernelForge Champion** | **cuBLAS GEMM (`torch.mm`) + fused Triton epilogue** | **1.573 ms** | **1.125x faster** | **0.0078** | **PASSED** |
| **PyTorch `torch.compile`** | TorchInductor full-graph loop fusion | 1.303 ms | 1.358x faster | 0.0000 | Reference |

---

## 3. Engineering Insights & Autopsy

### Hybrid Architecture (cuBLAS GEMM + Triton Epilogue)

In **Experiment 1**, the agent attempted a from-scratch Triton GEMM kernel. It was numerically correct but ran at 46.06 ms (~23x slower than cuBLAS) due to unspecialized tile loads on Turing SM75 hardware.

For **Experiment 2**, the task was restructured into a practical hybrid pipeline:

1. **Compute-bound GEMM:** kept in hardware-tuned cuBLAS (`torch.mm`). The agent stacked the projection weights (`torch.cat([w1, w2], dim=0)`) into a single fused GEMM call to maximize Tensor Core occupancy.
2. **Bandwidth-bound epilogue:** written by the agent as a custom `@triton.jit` kernel (`_swiglu_fused_kernel`). It loads the GEMM output tiles into FP32 registers, adds biases, evaluates SiLU(t) = t / (1 + e^-t), and performs the elementwise multiply in a single pass before writing back to GDDR6 in FP16.

### Autonomous Self-Repair Evidence

In run `run_20261026_125246.jsonl`:

- **Iteration 1:** the generated kernel failed numerical tolerance (`max_abs_err = 14.453`) due to improper register precision and bias broadcasting.
- **Iteration 2:** after receiving the numerical feedback, Nemotron repaired the accumulator casting (`tl.float32`) and coordinate calculations, reaching `max_abs_err = 0.0078125` (within tolerance) and securing the **1.125x speedup (1.573 ms)**.
- **Iterations 3 & 4:** further tuning attempts drifted in correctness (`err = 16.646`). The stopping rule halted the loop after two non-improving iterations to preserve the champion.

### Why `torch.compile` Outperforms the Champion

TorchInductor reaches 1.303 ms because it lowers the whole graph across tensor stride metadata without intermediate allocation. The champion host wrapper instead calls `.contiguous()` on the output slices `Y[:, :N]` and `Y[:, N:]`, allocating and copying two intermediate `[4096, 2048]` FP16 matrices to global memory before launching the Triton epilogue. This documents the concrete boundary between elementwise kernel fusion and full graph-level memory lowering.

---

## 4. Repository Layout

| Path | Purpose |
|:---|:---|
| `agent.py` | Autonomous generate / benchmark / self-repair loop |
| `runner.py` | Subprocess harness: runs a candidate kernel, returns JSON result |
| `workload.py` | PyTorch reference workload (SwiGLU) |
| `benchmark.py` | Benchmark helpers |
| `champion_kernel.py`, `champion_epilogue.py` | Best kernels found by the agent |
| `task_epilogue.txt` | Task prompt for Experiment 2 |
| `results_epilogue.json` | Verified benchmark results (Tesla T4) |
| `run_20261026_125246.jsonl` | Full run log used for the autopsy above |
| `dashboard/` | Static live dashboard (GitHub Pages) |
| `backend/` | FastAPI backend serving run data |
| `frontend/` | React + Vite web frontend |
| `KernelForge_Nebius_run_v3.ipynb` | One-click Colab notebook |

---

## 5. Reproducibility & Benchmark Instructions

### Option A: 1-Click Execution via Google Colab

1. Open `KernelForge_Nebius_run_v3.ipynb` in Google Colab.
2. Change runtime: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells. When prompted, provide your `NEBIUS_API_KEY` (kept in memory via `getpass`, never saved to disk).
4. Cell 7 runs the agent loop; Cell 9 verifies results against eager PyTorch and `torch.compile` (producing `results_epilogue.json`).

### Option B: Local / Cloud GPU Shell Execution

```bash
# 1. Clone repository
git clone https://github.com/ruchitham2025-byte/KernelForge.git
cd KernelForge

# 2. Install dependencies (PyTorch with CUDA, Triton, and OpenAI client)
pip install torch triton openai

# 3. Configure Nebius API credentials
export NEBIUS_API_KEY="your-api-key-here"
export NEBIUS_MODEL="nvidia/nemotron-3-super-120b-a12b"

# 4. Run the autonomous optimization loop (Experiment 2: Fused Epilogue)
python agent.py --iters 8 --task-file task_epilogue.txt

# 5. Run the fair multi-baseline benchmark against eager PyTorch and torch.compile
python runner.py --final --file champion_epilogue.py
```

### Option C: Run the Web App Locally

```bash
# Backend (terminal 1)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend (terminal 2)
cd frontend
npm install
npm run dev
```
