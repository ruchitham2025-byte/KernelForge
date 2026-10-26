"""runner.py: runs ONE candidate Triton module against the PyTorch reference.

  python runner.py --baseline                -> times the PyTorch reference
  python runner.py < candidate.py            -> judge a candidate (code on stdin)
  python runner.py --final --file champ.py   -> fair final benchmark: eager vs torch.compile vs champion
                                                (writes results.json)

Always prints one JSON object as the LAST line of stdout.
"""
import importlib.util
import json
import math
import os
import sys
import time
import traceback

try:
    import torch
except ImportError:  # reported as JSON from main()
    torch = None

M, K, N = 4096, 1024, 2048
ATOL, RTOL = 5e-3, 1e-2


def emit(d):
    sys.stdout.flush()
    print(json.dumps(d), flush=True)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def make_inputs():
    g = torch.Generator().manual_seed(0)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to("cuda", torch.float16)

    s = 1.0 / math.sqrt(K)
    return r(M, K), r(N, K, scale=s), r(N, scale=0.1), r(N, K, scale=s), r(N, scale=0.1)


def reference(x, w1, b1, w2, b2):
    a = torch.matmul(x, w1.t()) + b1
    b = torch.matmul(x, w2.t()) + b2
    return a * torch.nn.functional.silu(b)


def bench(fn):
    import triton
    try:
        return float(triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median"))
    except TypeError:
        return float(triton.testing.do_bench(fn, warmup=50, rep=200))


def warm_up(fn, seconds=1.0):
    """Run fn for a while so the GPU clocks are up before anything is timed."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
    torch.cuda.synchronize()


def measure_rounds(fns, rounds, bench_fn):
    """Interleave the callables round by round so clock/thermal drift hits all of them equally."""
    times = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            times[k].append(bench_fn(f))
    return times


def load_candidate(code):
    path = os.path.abspath("candidate_kernel.py")
    with open(path, "w") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod


def check_candidate(fn, inputs):
    """Run the candidate once; return (ran, numerics_ok, max_abs_err)."""
    ref = reference(*inputs)
    res = fn(*inputs)
    torch.cuda.synchronize()
    if not isinstance(res, torch.Tensor):
        raise TypeError(f"triton_swiglu returned {type(res).__name__}, expected a torch.Tensor")
    if (not res.is_cuda) or tuple(res.shape) != (M, N) or res.dtype != torch.float16:
        raise ValueError(f"Expected CUDA float16 tensor of shape ({M}, {N}), "
                         f"got shape {tuple(res.shape)}, dtype {res.dtype}, device {res.device}")
    finite = bool(torch.isfinite(res).all().item())
    max_err = float((res.float() - ref.float()).abs().max().item()) if finite else None
    ok = finite and bool(torch.allclose(res.float(), ref.float(), rtol=RTOL, atol=ATOL))
    return ok, max_err


def main():
    args = sys.argv[1:]
    out = {"ran": False, "success": False, "numerics_ok": False, "latency_ms": None,
           "baseline_ms": None, "speedup": None, "max_abs_err": None,
           "error": "", "traceback": ""}
    try:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU available")
        inputs = make_inputs()
        eager = lambda: reference(*inputs)

        if "--baseline" in args:
            warm_up(eager)
            t = measure_rounds({"eager": eager}, 3, bench)["eager"]
            ms = median(t)
            out.update(ran=True, success=True, latency_ms=ms, baseline_ms=ms)
            emit(out)
            return

        if "--file" in args:
            with open(args[args.index("--file") + 1]) as f:
                code = f.read()
        else:
            code = sys.stdin.read()

        mod = load_candidate(code)
        fn = getattr(mod, "triton_swiglu", None)
        if not callable(fn):
            raise RuntimeError("Module does not define a callable triton_swiglu(x, w1, b1, w2, b2)")
        cand = lambda: fn(*inputs)

        ok, max_err = check_candidate(fn, inputs)
        out.update(ran=True, max_abs_err=max_err, numerics_ok=ok, success=ok)

        if "--final" in args:
            import triton
            fns = {"champion": cand, "eager": eager}
            compile_err = None
            try:
                compiled = torch.compile(reference)
                compiled(*inputs)
                torch.cuda.synchronize()
                fns["torch_compile"] = lambda: compiled(*inputs)
            except Exception as e:
                compile_err = f"{type(e).__name__}: {e}"[:300]
            for f in fns.values():
                warm_up(f)
            rounds = 7
            times = measure_rounds(fns, rounds, bench)
            med = {k: median(v) for k, v in times.items()}
            res = {
                "ran": True, "numerics_ok": ok, "max_abs_err": max_err,
                "champion_ms": med["champion"], "eager_ms": med["eager"],
                "torch_compile_ms": med.get("torch_compile"),
                "speedup_vs_eager": med["eager"] / med["champion"],
                "speedup_vs_compile": (med["torch_compile"] / med["champion"]) if "torch_compile" in med else None,
                "torch_compile_error": compile_err, "rounds": rounds,
                "runs_ms": times, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "triton": triton.__version__,
                "shapes": {"x": [M, K], "w1": [N, K], "w2": [N, K], "out": [M, N]},
                "tolerance": {"atol": ATOL, "rtol": RTOL},
                "method": "median of 7 interleaved rounds of triton.testing.do_bench after 1 s GPU warm-up each",
            }
            with open("results.json", "w") as f:
                json.dump(res, f, indent=2)
            emit(res)
            return

        if ok:
            warm_up(eager)
            warm_up(cand)
            times = measure_rounds({"cand": cand, "eager": eager}, 3, bench)
            t_c, t_b = median(times["cand"]), median(times["eager"])
            out.update(latency_ms=t_c, baseline_ms=t_b, speedup=t_b / t_c)
        emit(out)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        out["traceback"] = traceback.format_exc()[-1500:]
        emit(out)


if __name__ == "__main__":
    main()
