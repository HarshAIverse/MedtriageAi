"""validate.py - Smoke-test for Medical Triage Environment"""
import sys

sys.path.insert(0, ".")

# ── 1. Import ────────────────────────────────────────────────────────────────
from environment import (
    MedicalTriageEnv, Action, Observation, Reward, PATIENT_CASES
)
print(f"[OK] environment.py loaded - {len(PATIENT_CASES)} patient cases")

for c in PATIENT_CASES:
    print(f"     {c.patient_id} [{c.difficulty:6s}] ESI={c.correct_triage_level}  {c.history['chief_complaint']}")

# ── 2. reset() / step() ─────────────────────────────────────────────────────
env = MedicalTriageEnv(seed=42)
obs = env.reset(case_ids=["PT-001"], shuffle=False)
assert isinstance(obs, Observation)
print(f"\n[OK] reset()  patient={obs.patient_id}  HR={obs.vitals.heart_rate}  GCS={obs.vitals.glasgow_coma_scale}")

action = Action(
    triage_level=4,
    recommended_actions=["wound cleaning", "suture closure", "tetanus assessment"],
    critical_flags=[],
    reasoning="Minor laceration, normal vitals.",
)
_, reward, done, info = env.step(action)
assert 0.0 <= reward.score <= 1.0, f"Score out of range: {reward.score}"
assert done, "Episode should be done after 1 case"
print(f"[OK] step()   score={reward.score:.4f}  done={done}")
print(f"     breakdown={reward.breakdown}")

# ── 3. All reward scores in [0,1] ───────────────────────────────────────────
env2 = MedicalTriageEnv(seed=0)
obs2 = env2.reset(shuffle=False)
scores = []
done2 = False
while not done2:
    act = Action(triage_level=3, recommended_actions=["assessment", "IV access"], critical_flags=[])
    _, rew, done2, inf = env2.step(act)
    scores.append(rew.score)
    assert 0.0 <= rew.score <= 1.0, f"INVALID SCORE {rew.score} for {inf['case_id']}"

print(
    f"\n[OK] All {len(scores)} reward scores in [0,1]  "
    f"min={min(scores):.4f}  max={max(scores):.4f}  mean={sum(scores)/len(scores):.4f}"
)

# ── 4. state() ──────────────────────────────────────────────────────────────
env3 = MedicalTriageEnv()
env3.reset(case_ids=["PT-001", "PT-002"], shuffle=False)
st = env3.state()
print(f"\n[OK] state() keys: {list(st.keys())}")
print(f"     episode_length={st['episode_length']}  current_patient={st['current_patient_id']}")

# ── 5. Tasks graders ────────────────────────────────────────────────────────
from tasks import TASK_REGISTRY, run_all_tasks

def baseline_agent(obs: Observation) -> Action:
    hr = obs.vitals.heart_rate
    spo2 = obs.vitals.oxygen_saturation
    gcs = obs.vitals.glasgow_coma_scale
    sbp = obs.vitals.systolic_bp
    if gcs < 13 or spo2 < 90 or sbp < 90:
        level, flags, acts = 1, ["critical deterioration"], ["IV access", "oxygen", "emergency consult"]
    elif spo2 < 94 or hr > 120 or sbp < 100:
        level, flags, acts = 2, ["hemodynamic compromise"], ["IV access", "labs", "monitoring"]
    elif hr > 100 or obs.vitals.temperature_celsius > 38.5:
        level, flags, acts = 3, [], ["labs", "IV access", "assessment"]
    elif obs.vitals.pain_score > 5:
        level, flags, acts = 4, [], ["analgesia", "assessment"]
    else:
        level, flags, acts = 5, [], ["symptomatic treatment", "discharge"]
    return Action(triage_level=level, recommended_actions=acts, critical_flags=flags)

print("\n[OK] Running all 3 tasks with baseline agent...")
results = run_all_tasks(baseline_agent, verbose=False)
for tid, score in results.items():
    meta = TASK_REGISTRY[tid]
    status = "PASS" if score >= meta.target_score else "below-target (expected for baseline)"
    print(f"     {tid}: score={score:.4f}  target>={meta.target_score}  [{status}]")

# ── 6. case_ids_by_difficulty ───────────────────────────────────────────────
env4 = MedicalTriageEnv()
by_diff = env4.case_ids_by_difficulty
print(f"\n[OK] case_ids_by_difficulty: {dict((k, len(v)) for k, v in by_diff.items())}")

print("\n" + "=" * 60)
print("ALL VALIDATION CHECKS PASSED")
print("=" * 60)
