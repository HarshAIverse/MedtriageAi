"""
_show_results.py  - extracts and prints only key lines from inference_out.txt
"""
KEYWORDS = (
    "[START]", "[STEP]", "[END]", "[FINAL]",
    "FINAL RESULTS", "task_1", "task_2", "task_3",
    "score=", "Overall", "wall_time", "LLM call",
    "======", "PASS", "FAIL", "Error", "error", "Traceback",
    "API Base", "Model", "Token set",
)

lines = open("inference_out.txt", encoding="utf-8-sig", errors="replace").readlines()
print(f"Total output lines: {len(lines)}\n")
for i, line in enumerate(lines, 1):
    stripped = line.rstrip()
    if any(k in stripped for k in KEYWORDS):
        print(f"{i:3d}: {stripped}")
