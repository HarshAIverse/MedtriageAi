"""_read_results.py — reads inference_out.txt and prints key lines to stdout"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

KEYWORDS = ('[START]', '[STEP]', '[END]', 'FINAL RESULTS',
            'score=', 'Overall', 'wall_time', 'LLM call',
            'PASS', 'FAIL', 'API Base', 'Model ', 'Token set')

lines = open('inference_out.txt', encoding='utf-8-sig', errors='replace').readlines()
print(f"Total lines in output: {len(lines)}")
for i, l in enumerate(lines, 1):
    s = l.rstrip()
    if any(k in s for k in KEYWORDS):
        print(f"{i:3d}: {s}")
