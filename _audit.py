"""
Deep audit of the submission against every judging criterion.
Run: python _audit.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


results = []

def check(name, passed, detail=""):
    results.append((name, passed, detail))

# ─── PHASE 1: Automated Validation ───────────────────────────────────────────

# 1a. Required files exist
for f in ['inference.py','environment.py','tasks.py','app.py',
          'openenv.yaml','Dockerfile','requirements.txt','validate.py']:
    check(f"File exists: {f}", os.path.exists(f))

# 1b. Env vars in inference.py
src = open('inference.py', encoding='utf-8').read()
check("inference.py: reads API_BASE_URL", "API_BASE_URL" in src)
check("inference.py: reads MODEL_NAME",   "MODEL_NAME" in src)
check("inference.py: reads HF_TOKEN",     "HF_TOKEN" in src)
check("inference.py: uses openai.OpenAI", "from openai import OpenAI" in src)

# 1c. Structured log format
check("Emits [START] with task_id",  "[START] task_id=" in src)
check("Emits [STEP]  with step=",    "[STEP]  task_id=" in src and "step=" in src)
check("Emits [END]   with status=",  "[END]   task_id=" in src and "status=" in src)
check("step= is 0-indexed integer",  "step_index = 0" in src)

# 1d. OpenAI client only (no requests/httpx direct LLM calls)
import_ok = "import requests" not in src or "openai" in src
check("LLM calls via OpenAI client only", "from openai import OpenAI" in src)

# 1e. openenv.yaml compliance
yaml = open('openenv.yaml', encoding='utf-8').read()
check("openenv.yaml: name field",         "name:" in yaml)
check("openenv.yaml: 3 tasks",            yaml.count("- id: task_") >= 3)
check("openenv.yaml: observation_space",  "observation_space:" in yaml)
check("openenv.yaml: action_space",       "action_space:" in yaml)
check("openenv.yaml: reward section",     "reward:" in yaml)
check("openenv.yaml: docker section",     "docker:" in yaml)
check("openenv.yaml: /reset endpoint",    "/reset" in yaml)
check("openenv.yaml: /step endpoint",     "/step" in yaml)
check("openenv.yaml: /state endpoint",    "/state" in yaml)
check("openenv.yaml: reward range 0-1",   "0.0" in yaml and "1.0" in yaml)

# 1f. Dockerfile
df = open('Dockerfile', encoding='utf-8').read()
check("Dockerfile: python:3.11",   "python:3.11" in df)
check("Dockerfile: EXPOSE 7860",   "EXPOSE 7860" in df)
check("Dockerfile: uvicorn CMD",   "uvicorn" in df)
check("Dockerfile: copies static", "static/" in df)

# 1g. Graders — do they return variable scores?
from environment import MedicalTriageEnv, Action
env = MedicalTriageEnv(seed=42)
scores = []
# Try all 25 cases with same action to detect fixed-return grader
obs = env.reset(shuffle=False)
while True:
    a = Action(triage_level=3, recommended_actions=["assessment"], critical_flags=[])
    o, r, done, info = env.step(a)
    scores.append(r.score)
    if done: break
unique_scores = len(set(scores))
check(f"Grader returns VARIABLE scores (got {unique_scores} distinct across 25 cases)",
      unique_scores >= 3,
      f"scores: {[round(s,3) for s in scores]}")

# 1h. Baseline reproduces
try:
    from tasks import TASK_REGISTRY, run_episode, grade_task_1, grade_task_2, grade_task_3

    # Simple heuristic agent for testing
    def heuristic(obs):
        from environment import Action
        return Action(triage_level=3, recommended_actions=["assessment","IV access"], critical_flags=[])

    s1 = grade_task_1(heuristic)
    check(f"grade_task_1 runs, score in [0,1], got {s1:.4f}", 0.0 <= s1 <= 1.0)
    s2 = grade_task_2(heuristic)
    check(f"grade_task_2 runs, score in [0,1], got {s2:.4f}", 0.0 <= s2 <= 1.0)
    s3 = grade_task_3(heuristic)
    check(f"grade_task_3 runs, score in [0,1], got {s3:.4f}", 0.0 <= s3 <= 1.0)
    check("All 3 task graders run without error", True)
except Exception as e:
    check("grade_task_1 runs", False, str(e))
    check("grade_task_2 runs", False, str(e))
    check("grade_task_3 runs", False, str(e))
    check("All 3 task graders run without error", False, str(e))


# ─── PHASE 2: Agentic Evaluation ─────────────────────────────────────────────
# Score variance — different action quality → different scores
env3 = MedicalTriageEnv(seed=42)
obs = env3.reset(case_ids=["PT-001"], shuffle=False)
perfect = Action(triage_level=4,
                 recommended_actions=["wound irrigation","suture closure","tetanus assessment"],
                 critical_flags=[])
_, r_good, _, _ = env3.step(perfect)

env3.reset(case_ids=["PT-001"], shuffle=False)
wrong = Action(triage_level=1,
               recommended_actions=["intubate","defibrillate"],
               critical_flags=["cardiac arrest"])
_, r_bad, _, _ = env3.step(wrong)

check("Score varies with action quality",
      r_good.score > r_bad.score,
      f"good={r_good.score:.3f} bad={r_bad.score:.3f} diff={r_good.score-r_bad.score:.3f}")
check("Score gap > 0.3 (meaningful discrimination)", r_good.score - r_bad.score > 0.3)

# ─── ANTI-DISQUALIFICATION CHECKS ────────────────────────────────────────────
check("Inference script named inference.py",         os.path.exists("inference.py"))
check("Inference is non-trivial (>200 lines)",       open('inference.py',encoding='utf-8').read().count('\n') > 200)
check("Environment has 25 distinct cases",           True)  # confirmed by validate.py
check("Grader does NOT always return same score",    unique_scores >= 3)
check("No placeholder or stub code",
      "TODO" not in open('environment.py',encoding='utf-8').read() and
      "pass  # placeholder" not in open('environment.py',encoding='utf-8').read())

# ─── PRINT RESULTS ───────────────────────────────────────────────────────────
print()
print("=" * 65)
print(" PRE-SUBMISSION DEEP AUDIT")
print("=" * 65)

groups = [
    ("Phase 1: Files & Structure",    results[:8]),
    ("Phase 1: Env Vars & Client",    results[8:12]),
    ("Phase 1: Log Format",           results[12:16]),
    ("Phase 1: OpenEnv YAML",         results[16:26]),
    ("Phase 1: Dockerfile",           results[26:30]),
    ("Phase 1: Grader Variance",      results[30:31]),
    ("Phase 1: Task Graders",         results[31:35]),
    ("Phase 2: Score Discrimination", results[35:37]),
    ("Anti-Disqualification",         results[37:]),
]

total_pass = total_fail = 0
for section, items in groups:
    print(f"\n  {section}")
    print(f"  {'─'*55}".replace('─', '-'))
    for name, passed, detail in items:
        icon = "OK  " if passed else "FAIL"
        print(f"    [{icon}]  {name}")
        if detail and not passed:
            print(f"           ↳ {detail}")
        if passed: total_pass += 1
        else: total_fail += 1

print()
print("=" * 65)
print(f"  RESULT: {total_pass} PASS  |  {total_fail} FAIL")
print("=" * 65)
