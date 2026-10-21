import torch
import triton.testing
from workload import get_realistic_inputs, swiglu_reference
from champion_kernel import triton_swiglu

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w1, b1, w2, b2 = get_realistic_inputs(device=device)
    ref_out = swiglu_reference(x, w1, b1, w2, b2)
    cand_out = triton_swiglu(x, w1, b1, w2, b2)
    torch.cuda.synchronize()

    max_err = float((cand_out - ref_out).abs().max().item())
    torch.testing.assert_close(cand_out, ref_out, rtol=1e-2, atol=1e-2)

    baseline_ms = float(triton.testing.do_bench(lambda: swiglu_reference(x, w1, b1, w2, b2)))
    triton_ms = float(triton.testing.do_bench(lambda: triton_swiglu(x, w1, b1, w2, b2)))
    speedup = baseline_ms / triton_ms

    print("=" * 60)
    print(" KERNELFORGE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Device                   : {torch.cuda.get_device_name(0)}")
    print(f"Baseline PyTorch Latency : {baseline_ms:.4f} ms")
    print(f"Triton Champion Latency  : {triton_ms:.4f} ms")
    print(f"Max Absolute Error       : {max_err:.8f}")
    print(f"Speedup Ratio            : {speedup:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
