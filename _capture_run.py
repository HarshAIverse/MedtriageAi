"""
_capture_run.py
Runs inference.py and saves stdout/stderr + return code to capture.json
"""
import subprocess, sys, os, json

env = os.environ.copy()
env["HF_TOKEN"]     = "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"
env["API_BASE_URL"] = "https://openrouter.ai/api/v1"
env["MODEL_NAME"]   = "openai/gpt-4o-mini"

print("Running inference.py ... (this takes ~2 min)", flush=True)
r = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
    cwd=r"d:\Github\Meta",
    env=env,
)

result = {
    "exit_code": r.returncode,
    "stdout_lines": r.stdout.split("\n"),
    "stderr_lines": r.stderr.split("\n")[-20:] if r.stderr.strip() else [],
}

with open("capture.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Done. Exit code: {r.returncode}")
print(f"Stdout lines: {len(result['stdout_lines'])}")
print("Saved to capture.json")
