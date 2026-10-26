# KernelForge: Autonomous Hardware-in-the-Loop GPU Kernel Synthesis

> **Nebius x NVIDIA Global AI Hackathon Submission**  
> **Model:** `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory[cite: 1]  
> **Hardware:** NVIDIA Tesla T4 (Turing, SM75, 16GB GDDR6)[cite: 1, 2]  
> **Environment:** PyTorch 2.11.0+cu130, Triton 3.6.0  

---

## 1. Overview
KernelForge is an autonomous agent that tasks an open NVIDIA model to write, evaluate, and repair OpenAI Triton GPU kernels directly on physical accelerator hardware[cite: 1, 13].

The harness catches compiler tracebacks and numerical drift, feeds error metrics back into the context window, and converges on verified, hardware-specialized kernel implementations[cite: 1, 2].

---

## 2. Benchmark Results & Baselines

All measurements reflect the **median of 7 interleaved rounds** using `triton.testing.do_bench` after a 1.0-second active GPU warm-up[cite: 10, 11].

* **Workload:** SwiGLU Epilogue (`out = (x @ W1.T + b1) * silu(x @ W2.T + b2)`)[cite: 10, 16]
* **Shapes:** `x (4096, 1024)`, `W1, W2 (2048, 1024)`, `b1, b2 (2048,)` (FP16)[cite: 10, 16]
* **Tolerance:** `atol=5e-3, rtol=1e-2` against PyTorch FP16 reference[cite: 10]

| Implementation | Median Latency | Relative Speedup | Max Abs Error | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch Eager** | 1.770 ms | 1.000x (Baseline) | 0.0000 | Reference |
| **KernelForge Champion (Triton)** | **1.573 ms** | **1.125x faster** | **0.0078** | **PASSED** |
| **PyTorch 2.0 `torch.compile`** | 1.303 ms | 1.358x faster | 0.0000 | Compiled Reference |

---

## 3. Engineering Insights

* **Autonomous Self-Repair:** In Iteration 1, the model encountered severe numerical error (`err=14.453`). With structured feedback, it self-healed in Iteration 2, lowering error to `0.0078` and locking in the 1.125x speedup[cite: 4, 13].
* **Why Champion Beats Eager:** Concatenates dual projections into a single cuBLAS GEMM (`torch.mm(x, W.t())`) and fuses bias addition, SiLU, and Hadamard multiplication into a single-pass Triton kernel in FP32 registers[cite: 16].
* **Why `torch.compile` Leads:** TorchInductor executes loop fusion across tensor stride metadata without intermediate allocation. Our champion called `.contiguous()` on matrix slices, incurring global GDDR6 memory round-trips[cite: 2].

---

## 4. How to Reproduce

### Google Colab
1. Open `KernelForge_Nebius_run_v3.ipynb` in Colab[cite: 6].
2. Select **Runtime > Change runtime type > T4 GPU**[cite: 8].
3. Run All Cells and provide your `NEBIUS_API_KEY`[cite: 8].
4. The loop writes `results_epilogue.json` and outputs run logs[cite: 8].

### Local CLI
```bash
export NEBIUS_API_KEY="your_api_key"
export NEBIUS_MODEL="nvidia/nemotron-3-super-120b-a12b"

# Run autonomous agent loop
python agent.py --iters 8 --task-file task_epilogue.txt

# Run benchmark suite
python runner.py --final --file champion_kernel.py
