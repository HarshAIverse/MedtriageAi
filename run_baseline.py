"""
run_baseline.py  --  Runs the deterministic baseline agent through all 3 tasks
and prints a clean, machine-readable per-step and summary report.
"""
import sys
sys.path.insert(0, ".")

from environment import MedicalTriageEnv, Action, Observation
from tasks import TASK_REGISTRY, run_episode

# ── Baseline heuristic agent ──────────────────────────────────────────────────
def baseline_agent(obs: Observation) -> Action:
    hr   = obs.vitals.heart_rate
    spo2 = obs.vitals.oxygen_saturation
    gcs  = obs.vitals.glasgow_coma_scale
    sbp  = obs.vitals.systolic_bp
    temp = obs.vitals.temperature_celsius
    pain = obs.vitals.pain_score

    if gcs < 13 or spo2 < 90 or sbp < 90:
        level = 1
        flags = ["critical deterioration"]
        acts  = ["IV access x2", "oxygen 100%", "emergency consult", "continuous monitoring"]
    elif spo2 < 94 or hr > 120 or sbp < 100:
        level = 2
        flags = ["hemodynamic compromise"]
        acts  = ["IV access", "labs", "cardiac monitoring", "specialist consult"]
    elif hr > 100 or temp > 38.5:
        level = 3
        flags = []
        acts  = ["labs", "IV access", "clinical assessment"]
    elif pain > 5:
        level = 4
        flags = []
        acts  = ["analgesia", "assessment", "targeted workup"]
    else:
        level = 5
        flags = []
        acts  = ["symptomatic treatment", "discharge with follow-up"]

    return Action(
        triage_level=level,
        recommended_actions=acts,
        critical_flags=flags,
        reasoning="Vitals-threshold heuristic.",
    )


# ── Run each task, print per-case table ──────────────────────────────────────
task_grades = {}

for task_id, meta in TASK_REGISTRY.items():
    result = run_episode(
        agent=baseline_agent,
        case_ids=meta.case_ids,
        shuffle=False,
        seed=42,
    )

    # Task-specific grading
    if task_id == "task_1":
        grade = round(max(0.0, min(1.0, result["mean_score"])), 4)

    elif task_id == "task_2":
        ambiguous = {"PT-009", "PT-011"}
        case_scores = {info["case_id"]: s for info, s in zip(result["info_list"], result["scores"])}
        weights = {cid: (1.5 if cid in ambiguous else 1.0) for cid in meta.case_ids}
        num = sum(case_scores.get(cid, 0.0) * w for cid, w in weights.items())
        den = sum(weights.values())
        grade = round(max(0.0, min(1.0, num / den)), 4)

    else:  # task_3
        base = result["mean_score"]
        adj = 0.0
        for info, act in zip(result["info_list"], result["actions"]):
            gt = info["correct_critical_flags"]
            pred = act.get("critical_flags", [])
            if gt:
                gt_tok = set(" ".join(f.lower() for f in gt).split())
                pred_tok = set(" ".join(f.lower() for f in pred).split())
                recall = len(gt_tok & pred_tok) / max(len(gt_tok), 1)
                if recall == 1.0:
                    adj += 0.05
                elif recall == 0.0:
                    adj -= 0.05
        grade = round(max(0.0, min(1.0, base + adj)), 4)

    task_grades[task_id] = grade

    # Per-case table
    sep = "-" * 78
    print(f"\n{sep}")
    print(f"TASK {task_id[-1]}  {meta.name}  [{meta.difficulty.upper()}]  target >= {meta.target_score}")
    print(sep)
    hdr = f"{'Patient':<10} {'Assigned':>9} {'Correct':>8} {'Score':>7}  Flags Caught"
    print(hdr)
    print("-" * 78)
    for info, act, score in zip(result["info_list"], result["actions"], result["scores"]):
        flag_hit = "yes" if act["critical_flags"] and info["correct_critical_flags"] else (
                   "n/a" if not info["correct_critical_flags"] else "NO")
        print(
            f"{info['case_id']:<10} {act['triage_level']:>9} {info['correct_triage_level']:>8} "
            f"{score:>7.4f}  {flag_hit}"
        )
    print(sep)
    status = "PASS" if grade >= meta.target_score else "FAIL"
    print(f"Grade: {grade:.4f}  [{status}]  (wall_time={result['wall_time_s']:.3f}s)")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("BASELINE AGENT FINAL SUMMARY")
print("=" * 78)
overall = sum(task_grades.values()) / len(task_grades)
for task_id, grade in task_grades.items():
    meta = TASK_REGISTRY[task_id]
    status = "PASS" if grade >= meta.target_score else "FAIL"
    bar = "#" * int(grade * 20) + "." * (20 - int(grade * 20))
    print(f"  {task_id}  [{meta.difficulty:6s}]  [{bar}]  {grade:.4f}  {status}  (target >= {meta.target_score})")
print(f"\n  Overall mean : {overall:.4f}")
print("=" * 78)
print("Note: Baseline uses only vitals thresholds. LLM agents score significantly higher.")
