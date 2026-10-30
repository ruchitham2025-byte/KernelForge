"""KernelForge agent: asks a Nemotron model (Nebius Token Factory) for a Triton SwiGLU kernel,
runs it through runner.py, feeds the real result/traceback back, and keeps the fastest correct kernel."""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
FENCE = "`" * 3

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer who writes OpenAI Triton kernels. "
    "Reply with exactly ONE " + FENCE + "python code block containing a complete, self-contained module "
    "(all imports included) and no other text."
)

HINTS = (
    "\n\nTriton notes: tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) must be tl.constexpr kernel parameters passed at "
    "launch, and tl.arange needs constexpr bounds. Use tl.dot on 2-D tiles accumulating in float32. Load the "
    "weight tile as [BLOCK_K, BLOCK_N] with pointer arithmetic instead of transposing. Use a 2-D launch grid."
)

# ---------- security validation (sandbox) ----------
DISALLOWED_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib"}


def validate_candidate_code(code_str: str) -> tuple[bool, str]:
    """Basic static analysis to reject unsafe imports or commands before running in subprocess."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError during AST parsing: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in DISALLOWED_MODULES:
                    return False, f"Unsafe import detected: '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in DISALLOWED_MODULES:
                    return False, f"Unsafe import detected: '{node.module}'"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__", "compile"}:
                return False, f"Disallowed function call detected: '{node.func.id}'"

    return True, "Code passed security check."


# ---------- pure helpers (unit-testable) ----------
def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    f = re.escape(FENCE)
    blocks = re.findall(f + r"(?:python|py)?[ \t]*\n(.*?)" + f, text, flags=re.S)
    blocks = [b for b in blocks if "triton_swiglu" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    if FENCE not in text and "def triton_swiglu" in text:
        return text.strip() + "\n"
    return None


def classify(res):
    if res.get("timeout"):
        return "timeout"
    if not res.get("ran"):
        return "compile_or_runtime_error"
    if not res.get("numerics_ok"):
        return "wrong_numerics"
    sp = res.get("speedup")
    return "correct_faster" if (sp is not None and sp > 1.0) else "correct_slower"


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def build_feedback(outcome, res, best_ms, atol, rtol):
    tail = (res.get("traceback") or res.get("error") or "")[-1500:]
    ask = "Reply with the full corrected module in ONE python code block and nothing else."
    if outcome in ("compile_or_runtime_error", "timeout", "security_rejection"):
        return f"Your code failed ({outcome}). Error output (end of traceback):\n{FENCE}\n{tail}\n{FENCE}\n{ask}"
    if outcome == "wrong_numerics":
        return (f"Your code ran but the result is wrong: max_abs_err={fmt(res.get('max_abs_err'), '.4g')} "
                f"(allowed: atol {atol} + rtol {rtol} against the PyTorch fp16 reference; None means NaN/inf in the output). {ask}")
    base = fmt(res.get("baseline_ms"))
    if outcome == "correct_slower":
        return (f"Correct (max_abs_err={fmt(res.get('max_abs_err'), '.4g')}) but not faster: {fmt(res.get('latency_ms'))} ms "
                f"vs PyTorch {base} ms ({fmt(res.get('speedup'), '.2f')}x). Best correct so far: {fmt(best_ms)} ms. "
                f"Try different BLOCK_M/BLOCK_N/BLOCK_K, num_warps, num_stages or program ordering. {ask}")
    return (f"Correct and faster: {fmt(res.get('latency_ms'))} ms vs PyTorch {base} ms ({fmt(res.get('speedup'), '.2f')}x). "
            f"Best correct so far: {fmt(best_ms)} ms. Can you make it faster while staying correct? {ask}")


def run_runner(args, code="", timeout=240):
    try:
        p = subprocess.run([sys.executable, "runner.py"] + args, input=code,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ran": False, "timeout": True, "traceback": f"Timed out after {timeout} s (compile or run too slow)."}
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    tail = (p.stderr or p.stdout or "")[-1500:]
    return {"ran": False, "traceback": f"runner.py produced no JSON (exit code {p.returncode}). Output tail:\n{tail}"}


def ask_model(client, model, messages):
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.3, max_tokens=8192)
            u = getattr(r, "usage", None)
            return (r.choices[0].message.content or ""), (getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))
        except Exception as e:  # noqa
            last = e
            print(f"  API error (attempt {attempt + 1}/3): {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Nebius API failed after 3 attempts: {last}")


def last_line(s):
    lines = [l.strip() for l in (s or "").strip().splitlines() if l.strip()]
    return lines[-1][:60] if lines else ""


# ---------- main loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--hints", action="store_true", help="add short Triton API notes to the task prompt")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--task", required=True, help="read the task from a json file")
    a = ap.parse_args()

    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        sys.exit("Error: set the NEBIUS_API_KEY environment variable.")
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=key, timeout=300)

    if a.list_models:
        for mid in sorted(m.id for m in client.models.list().data):
            print(mid)
        return

    model = a.model or os.environ.get("NEBIUS_MODEL")
    if not model:
        sys.exit("Error: no model. Set NEBIUS_MODEL or pass --model.")

    task_path = a.task
    with open(task_path) as f:
        task_cfg = json.load(f)

    # Resolve prompt_file relative to the task json's directory
    prompt_file = task_cfg["prompt_file"]
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(task_path)), prompt_file)
    with open(prompt_path) as f:
        task_prompt = f.read().strip()

    # Tolerance from task json
    atol = task_cfg["atol"]
    rtol = task_cfg["rtol"]

    os.makedirs("artifacts", exist_ok=True)
    run_id = f"run-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = os.path.join("artifacts", run_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.json")
    
    # Compute sha256 of the task file
    import hashlib
    with open(task_path, "rb") as f:
        task_sha256 = hashlib.sha256(f.read()).hexdigest()

    # First line of run.jsonl: header
    run_jsonl_path = os.path.join(run_dir, "run.jsonl")
    with open(run_jsonl_path, "w") as f:
        f.write(f'{{"header": true, "task_name": "{task_cfg["name"]}", "task_file_sha256": "{task_sha256}", "run_id": "{run_id}", "model": "{model}", "base_url": "{BASE_URL}"}}\n')

    log_path = os.path.join(run_dir, "run.jsonl")

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    base = run_runner(["--baseline", "--task", a.task])
    if not base.get("ran"):
        sys.exit("Baseline failed:\n" + (base.get("traceback") or base.get("error") or "unknown"))
    baseline_ms = base["latency_ms"]
    print(f"Model: {model}\nEndpoint: {BASE_URL}\nHints: {a.hints}\nInitial PyTorch timing: {baseline_ms:.3f} ms (separate process, cold GPU; NOT used for speedups)")
    log({"header": True, "task_name": task_cfg["name"], "task_file_sha256": task_sha256, "run_id": run_id, "model": model, "base_url": BASE_URL})

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt + (HINTS if a.hints else "")}]
    best = None  # (latency_ms, code, iteration, res)
    no_improve = 0
    rows = []

    for it in range(1, a.iters + 1):
        print(f"\n=== Iteration {it}/{a.iters} ===")
        try:
            text, (pt, ct) = ask_model(client, model, messages)
        except RuntimeError as e:
            log({"api_error": str(e), "iteration_failed": it})
            sys.exit(str(e))
        messages.append({"role": "assistant", "content": text})

        code = extract_code(text)
        if code is None:
            res, outcome = {"ran": False, "traceback": "No python code block containing triton_swiglu was found in your reply."}, "no_code"
            feedback = "I could not find a python code block defining triton_swiglu. " \
                       "Reply with ONE python code block containing the full module and nothing else."
        else:
            is_safe, sec_msg = validate_candidate_code(code)
            if not is_safe:
                res = {"ran": False, "traceback": f"Security Policy Violation: {sec_msg}"}
                outcome = "security_rejection"
                feedback = f"Your code was rejected by the security sandbox: {sec_msg}. Do not use unapproved system modules or dynamic evaluation. Reply with ONE python code block containing a pure Triton/PyTorch kernel."
            else:
                res = run_runner(["--task", a.task], code)
                outcome = classify(res)
                feedback = None

        lat = res.get("latency_ms")
        if outcome.startswith("correct") and lat is not None:
            if best is None or lat < best[0]:
                best, no_improve = (lat, code, it, res), 0
            else:
                no_improve += 1
        elif best is not None:
            no_improve += 1

        if feedback is None:
            feedback = build_feedback(outcome, res, best[0] if best else None, atol, rtol)
        messages.append({"role": "user", "content": feedback})

        err_tail = (res.get("traceback") or res.get("error") or "")[-1500:] if not outcome.startswith("correct") else ""
        rec = {"iteration": it, "timestamp": datetime.now().isoformat(timespec="seconds"), "model": model,
               "base_url": BASE_URL, "prompt_tokens": pt, "completion_tokens": ct, "code": code,
               "outcome": outcome, "latency_ms": lat, "baseline_ms": res.get("baseline_ms"),
               "speedup": res.get("speedup"), "max_abs_err": res.get("max_abs_err"), "error_tail": err_tail,
               "run_id": run_id, "atol": atol, "rtol": rtol}
        log(rec)
        rows.append(rec)
        sp = res.get("speedup")
        print(f"Iteration {it}: {outcome} | latency={fmt(lat)} ms | speedup={'n/a' if sp is None else format(sp, '.2f') + 'x'} | "
              f"err={fmt(res.get('max_abs_err'), '.4g')}")
        if best is not None and no_improve >= 2:
            print("No improvement for 2 iterations: stopping.")
            break

    final = None
    if best:
        champion_path = os.path.join(run_dir, "champion.py")
        with open(champion_path, "w") as f:
            f.write(best[1])
        print(f"\nChampion saved to {champion_path}. Running fair benchmark: eager vs torch.compile vs champion, "
              "7 interleaved rounds (torch.compile can take a minute)...")
        final = run_runner(["--final", "--task", a.task, "--file", champion_path, "--out", results_path], timeout=900)
    summary = {"summary": True, "champion_iteration": best[2] if best else None,
               "loop_baseline_ms": best[3].get("baseline_ms") if best else None,
               "loop_best_ms": best[0] if best else None,
               "loop_speedup": best[3].get("speedup") if best else None}
    if final and final.get("ran"):
        for k in ("eager_ms", "torch_compile_ms", "champion_ms", "speedup_vs_eager",
                  "speedup_vs_compile", "torch_compile_error"):
            summary[k] = final.get(k)
    if final and final.get("ran"):
        for k in ("eager_ms", "torch_compile_ms", "champion_ms", "speedup_vs_eager",
                  "speedup_vs_compile", "torch_compile_error"):
            summary[k] = final.get(k)

    log(summary)
    if best:
        if final and final.get("ran") and final.get("champion_ms"):
            tc = final.get("torch_compile_ms")
            tc_s = "n/a" if tc is None else f"{tc:.3f} ms"
            sc = final.get("speedup_vs_compile")
            sc_s = "n/a" if sc is None else f"{sc:.2f}x"
            print(f"\nChampion: iteration {best[2]} (saved to {champion_path}, details in {results_path})")
            print(f"  eager PyTorch   : {final['eager_ms']:.3f} ms")
            print(f"  torch.compile   : {tc_s}")
            print(f"  champion kernel : {final['champion_ms']:.3f} ms")
            print(f"  speedup vs eager: {final['speedup_vs_eager']:.2f}x | vs torch.compile: {sc_s}")
            if tc is None:
                print(f"  (torch.compile baseline unavailable: {final.get('torch_compile_error')})")
        else:
            why = last_line((final or {}).get("traceback") or (final or {}).get("error") or "unknown")
            print(f"\nChampion: iteration {best[2]}, {best[0]:.3f} ms, in-loop speedup "
                  f"{fmt(best[3].get('speedup'), '.2f')}x. Final benchmark failed: {why}")
    else:
        print("\nNo correct kernel found within the iteration limit.")

    print("\n=== Optimization Autopsy Table ===")
    print(f"{'Iter':<5}{'Outcome':<26}{'Latency(ms)':<13}{'Speedup':<9}{'MaxAbsErr':<11}Last error line")
    print("-" * 100)
    for r in rows:
        print(f"{r['iteration']:<5}{r['outcome']:<26}{fmt(r['latency_ms']):<13}{fmt(r['speedup'], '.2f'):<9}"
              f"{fmt(r['max_abs_err'], '.4g'):<11}{last_line(r['error_tail'])}")
    print(f"\nFull log: {log_path}")


if __name__ == "__main__":
    main()
