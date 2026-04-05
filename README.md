---
title: Medical Triage AI — OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: RL environment where an AI agent acts as an ED triage nurse
---

# 🏥 Medical Triage AI — OpenEnv Submission

> **OpenEnv Hackathon** | Author: [HarshAIverse](https://huggingface.co/HarshAIverse) | v1.0.0 | MIT

An RL environment where an AI agent acts as an **emergency department triage nurse**. The agent reads patient vitals, symptoms, and medical history, then makes structured clinical decisions — assigning ESI priority levels, recommending interventions, and flagging life-threatening conditions.

---

## ⚡ Quick Start

```bash
# Clone
git clone https://huggingface.co/spaces/HarshAIverse/medical-triage-env
cd medical-triage-env

# Install
pip install -r requirements.txt

# Set required environment variables
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="openai/gpt-4o-mini"
export HF_TOKEN="sk-or-v1-..."

# Run inference  →  produces structured [START]/[STEP]/[END] logs
python inference.py

# Or launch the interactive dashboard
uvicorn app:app --host 0.0.0.0 --port 7860
# Open → http://localhost:7860
```

**Windows PowerShell:**
```powershell
$env:API_BASE_URL = "https://openrouter.ai/api/v1"
$env:MODEL_NAME   = "openai/gpt-4o-mini"
$env:HF_TOKEN     = "sk-or-v1-..."
python inference.py
```

---

## 🔑 Required Environment Variables

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible API base URL | `https://openrouter.ai/api/v1` |
| `MODEL_NAME` | Model identifier | `openai/gpt-4o-mini` |
| `HF_TOKEN` | API key / HuggingFace token | `sk-or-v1-...` |

> These variables **must** be set before running `inference.py`. The script will exit with a clear error if any are missing.

---

## 📋 Inference Log Format (OpenEnv Spec)

`inference.py` emits structured stdout logs in the exact OpenEnv format:

```
======================================================================
Medical Triage Environment - OpenEnv Inference Runner
======================================================================
API Base URL : https://openrouter.ai/api/v1
Model        : openai/gpt-4o-mini
Token set    : YES

[START] task_id=task_1 name='Stable Patient Triage' difficulty=easy target=0.6 cases=['PT-001', 'PT-002', 'PT-003', 'PT-004', 'PT-005']
[STEP]  task_id=task_1 step=0 patient=PT-001 score=0.7783 assigned_level=4 correct_level=4
[STEP]  task_id=task_1 step=1 patient=PT-002 score=0.7167 assigned_level=4 correct_level=4
[STEP]  task_id=task_1 step=2 patient=PT-003 score=0.4962 assigned_level=4 correct_level=5
[STEP]  task_id=task_1 step=3 patient=PT-004 score=0.4667 assigned_level=4 correct_level=5
[STEP]  task_id=task_1 step=4 patient=PT-005 score=0.7000 assigned_level=5 correct_level=5
[END]   task_id=task_1 score=0.6316 target=0.6 status=PASS wall_time_s=17.12

[START] task_id=task_2 name='Mixed Urgency Triage' difficulty=medium target=0.4 cases=['PT-006', 'PT-008', 'PT-009', 'PT-010', 'PT-011']
[STEP]  task_id=task_2 step=0 patient=PT-006 score=0.7333 assigned_level=2 correct_level=2
[STEP]  task_id=task_2 step=1 patient=PT-008 score=0.7216 assigned_level=3 correct_level=3
[STEP]  task_id=task_2 step=2 patient=PT-009 score=0.3306 assigned_level=2 correct_level=2
[STEP]  task_id=task_2 step=3 patient=PT-010 score=0.1144 assigned_level=2 correct_level=3
[STEP]  task_id=task_2 step=4 patient=PT-011 score=0.3959 assigned_level=2 correct_level=2
[END]   task_id=task_2 score=0.4432 target=0.4 status=PASS wall_time_s=21.61

[START] task_id=task_3 name='Complex Multi-System Critical Triage' difficulty=hard target=0.2 cases=['PT-012', 'PT-014', 'PT-015', 'PT-019', 'PT-024']
[STEP]  task_id=task_3 step=0 patient=PT-012 score=0.3726 assigned_level=1 correct_level=1
[STEP]  task_id=task_3 step=1 patient=PT-014 score=0.1005 assigned_level=2 correct_level=1
[STEP]  task_id=task_3 step=2 patient=PT-015 score=0.1189 assigned_level=2 correct_level=1
[STEP]  task_id=task_3 step=3 patient=PT-019 score=0.0388 assigned_level=2 correct_level=1
[STEP]  task_id=task_3 step=4 patient=PT-024 score=0.4022 assigned_level=2 correct_level=2
[END]   task_id=task_3 score=0.2066 target=0.2 status=PASS wall_time_s=25.95

======================================================================
FINAL RESULTS
======================================================================
  task_1  [easy  ]  score=0.6316  target>=0.6  [OK] PASS
  task_2  [medium]  score=0.4432  target>=0.4  [OK] PASS
  task_3  [hard  ]  score=0.2066  target>=0.2  [OK] PASS

  Overall mean score : 0.4271
  Total wall time    : 64.7s
  LLM call stats     : {'total_calls': 15, 'total_latency_s': 64.69, 'avg_latency_s': 4.31}
======================================================================
```

---

## 🎯 Real-World Motivation

ED triage errors cause **preventable deaths** and waste limited resources. The ESI (Emergency Severity Index) triage system is the gold standard, but even trained nurses make errors — particularly when patients present with atypical symptoms or misleading vitals:

- An elderly patient maintaining normal BP despite active haemorrhage
- Complete heart block presenting as "anxiety"
- Meningococcal septicaemia with only subtle early petechiae
- Digoxin toxicity in a patient who appears well and mobile

This environment trains AI agents to triage 25 medically accurate synthetic cases, from trivial lacerations to aortic dissections, with partial-credit rewards that reward vigilance without encouraging over-triage of stable patients.

---

## 📐 Tasks

### Task 1 — Stable Patient Triage `[easy]` · Target ≥ 0.60

**Cases:** PT-001, PT-002, PT-003, PT-004, PT-005

5 patients with clearly benign presentations: small laceration, ankle sprain (Ottawa criteria positive), strep pharyngitis, mechanical back pain, viral URTI. All vitals normal. The agent must assign **ESI 4–5** and recommend appropriate outpatient actions.

> **Key pitfall:** Over-triaging these patients (assigning ESI 1–2) incurs a −0.10 penalty per case.

---

### Task 2 — Mixed Urgency Triage `[medium]` · Target ≥ 0.40

**Cases:** PT-006, PT-008, PT-009 *(1.5×)*, PT-010, PT-011 *(1.5×)*

5 cases of mixed acuity. Ambiguous cases are weighted 1.5×:

| Patient | Condition | Correct ESI | Pitfall |
|---|---|---|---|
| PT-006 | Classic ACS — 67M, crushing chest pain, diaphoresis | 2 | Must not be labelled ESI 3 |
| PT-008 | Renal colic, morphine allergy | 3 | Allergy-aware analgesia required |
| PT-009 ⭐ | Urosepsis in elderly diabetic — GCS drop | 2 | Deceptively "moderate" vitals |
| PT-010 | Appendicitis — Alvarado score 7 | 3 | Needs CT + surgical consult |
| PT-011 ⭐ | Thunderclap headache / SAH rule-out | 2 | Must not be dismissed as migraine |

---

### Task 3 — Complex Multi-System Critical `[hard]` · Target ≥ 0.20

**Cases:** PT-012, PT-014, PT-015, PT-019, PT-024

5 cases engineered to deceive pattern-matching agents with misleading presentations:

| Patient | Condition | Hidden Trap |
|---|---|---|
| PT-012 | Opioid overdose | Partial naloxone response — airway still at risk |
| PT-014 | Acute ischemic stroke | On apixaban — tPA protocol complicated |
| PT-015 | 3rd-degree heart block | Patient attributes symptoms to "anxiety"; HR 38 |
| PT-019 | Meningococcal septicaemia | Penicillin anaphylaxis — must choose ceftriaxone |
| PT-024 | Digoxin toxicity | Appears well and mobile — HR 44, nausea dismissed |

> **Flag miss penalty:** −0.05 per case where critical flags are entirely absent from the prediction.

---

## 📊 Baseline Score Comparison

| Task | Naive Heuristic | GPT-4o-mini | GPT-4o | Target |
|---|---|---|---|---|
| Task 1 — Easy | ~0.19 | **0.63** | ~0.88 | ≥ 0.60 |
| Task 2 — Medium | ~0.08 | **0.44** | ~0.76 | ≥ 0.40 |
| Task 3 — Hard | ~0.00 | **0.21** | ~0.62 | ≥ 0.20 |
| **Overall mean** | **~0.09** | **0.43** | **~0.75** | — |

> - *Naive heuristic* uses only SpO2/HR/BP/GCS thresholds — no clinical reasoning
> - *GPT-4o-mini scores* are **measured** at temperature=0, 15 API calls total (~65s)
> - *GPT-4o scores* are estimated from a single validation run

---

## 💡 Reward Function

```
score = clamp(
    0.50 × triage_level_accuracy     # 1.0 exact | 0.5 for ±1 level | 0.0 for ≥2 off
  + 0.20 × action_concept_coverage   # synonym + bigram aware (not strict keyword match)
  + 0.20 × critical_flag_recall      # fraction of ground-truth flags identified
  + 0.10 × flag_recall_bonus         # bonus when critical flags are caught
  - 0.30 × missing_flag_fraction     # penalty when critical flags entirely missed
  - 0.10 × overtriage_penalty        # penalty: ESI 1–2 assigned to ESI 4–5 patient
, 0.0, 1.0)
```

**Design decisions:**
- **Concept-aware action scorer**: Uses synonym groups and bigram matching so "supplemental oxygen" and "O2 therapy" both score correctly — LLM paraphrasing doesn't cost points unfairly
- **Robust JSON coercion**: Handles string triage levels (`"4"`), null fields, comma-separated lists from any LLM output style
- **Over-triage penalty**: Explicitly penalises assigning ESI 1–2 to clearly stable patients — important for clinical safety

---

## 🔌 API Reference

The FastAPI server exposes these endpoints. The interactive dashboard is at `GET /`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive triage dashboard (HTML) |
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `POST` | `/reset` | Start episode. Body: `{"case_ids": [...], "shuffle": false}` |
| `POST` | `/step` | Submit action. Body: `{"action": {triage_level, recommended_actions, critical_flags}}` |
| `GET` | `/state` | Full environment state dict |
| `GET` | `/cases` | All 25 case IDs grouped by difficulty |
| `GET` | `/case/{patient_id}` | Metadata for a specific patient case |
| `GET` | `/tasks` | Task definitions with targets and case lists |
| `POST` | `/run_task` | Run a full task with built-in heuristic baseline agent |
| `GET` | `/docs` | Swagger interactive API documentation |

### Example: Reset + Step

```bash
# Start Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"case_ids": ["PT-001","PT-002","PT-003","PT-004","PT-005"], "shuffle": false}'

# Submit triage for PT-001 (minor laceration)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "triage_level": 4,
      "recommended_actions": ["wound irrigation", "suture closure", "tetanus assessment"],
      "critical_flags": [],
      "reasoning": "Minor laceration, all vitals normal, no systemic involvement."
    }
  }'
```

**Step response:**
```json
{
  "observation": { "patient_id": "PT-002", "vitals": { "heart_rate": 68, ... }, "symptoms": [...] },
  "reward": {
    "score": 0.7783,
    "breakdown": {
      "triage_level_score": 1.0,
      "action_coverage_score": 0.45,
      "flag_recall_score": 1.0,
      "critical_flag_bonus": 0.0,
      "missing_flag_penalty": -0.0,
      "overtriage_penalty": -0.0,
      "total": 0.7783
    },
    "feedback": "Correct ESI level. Good action coverage."
  },
  "done": false,
  "info": { "case_id": "PT-001", "correct_triage_level": 4, "correct_critical_flags": [], "difficulty": "easy" }
}
```

---

## 🗂️ Observation Space

| Field | Type | Description |
|---|---|---|
| `patient_id` | `str` | Unique case ID (e.g. `PT-012`) |
| `vitals.heart_rate` | `int` bpm | Heart rate |
| `vitals.systolic_bp` | `int` mmHg | Systolic blood pressure |
| `vitals.diastolic_bp` | `int` mmHg | Diastolic blood pressure |
| `vitals.respiratory_rate` | `int` br/min | Respiratory rate |
| `vitals.temperature_celsius` | `float` °C | Body temperature |
| `vitals.oxygen_saturation` | `float` % | SpO2 |
| `vitals.glasgow_coma_scale` | `int` [3–15] | GCS total score |
| `vitals.pain_score` | `int` [0–10] | Numeric pain scale |
| `symptoms` | `list[str]` | Presenting symptoms |
| `history.age` | `int` years | Patient age |
| `history.sex` | `str` M/F | Biological sex |
| `history.chief_complaint` | `str` | Primary ED presentation |
| `history.onset_hours` | `float` | Hours since symptom onset |
| `history.past_medical_history` | `list[str]` | Relevant diagnoses |
| `history.current_medications` | `list[str]` | Active medications |
| `history.allergies` | `list[str]` | Known drug allergies |
| `history.surgical_history` | `list[str]` | Past surgeries |
| `additional_context` | `str` | Nurse notes, labs, clinical scores (Alvarado, Wells PE, etc.) |
| `timestamp` | `str` ISO-8601 | Presentation time |
| `case_index` | `int` | 0-based position in episode |
| `total_cases` | `int` | Total cases in current episode |

---

## ⚙️ Action Space

| Field | Type | Constraint | Description |
|---|---|---|---|
| `triage_level` | `int` | 1–5 | ESI priority (1=Immediate → 5=Non-urgent) |
| `recommended_actions` | `list[str]` | ≥1 item | Ordered clinical interventions |
| `critical_flags` | `list[str]` | can be empty | Suspected life-threatening diagnoses |
| `reasoning` | `str` | optional | Clinical rationale / differential |

**ESI Reference:**

| Level | Label | Definition | Examples |
|---|---|---|---|
| **1** | Immediate | Life-saving intervention required *right now* | Cardiac arrest, GCS <8, airway obstruction |
| **2** | Emergent | High-risk — cannot safely wait | ACS, stroke, sepsis, ectopic pregnancy |
| **3** | Urgent | Stable, needs multiple resources | Pneumonia, appendicitis, fracture |
| **4** | Less Urgent | Stable, needs 1 resource | Sprain + X-ray, laceration repair, strep swab |
| **5** | Non-Urgent | Minor, 0 resources needed | Viral URTI, medication refill, minor complaint |

---

## 🐳 Docker

```bash
# Build
docker build -t medical-triage-env:latest .

# Run
docker run -d \
  --name triage-env \
  -p 7860:7860 \
  -e API_BASE_URL="https://openrouter.ai/api/v1" \
  -e MODEL_NAME="openai/gpt-4o-mini" \
  -e HF_TOKEN="sk-or-v1-..." \
  medical-triage-env:latest

# Health check
curl http://localhost:7860/health
# → {"status": "ok", "version": "1.0.0"}

# Run inference inside container
docker exec triage-env python inference.py
```

---

## 📁 File Structure

```
medical-triage-env/
├── environment.py     # Core RL env: Pydantic models, 25 patient cases, reward function
├── tasks.py           # TASK_REGISTRY: 3 graded tasks with individual graders
├── inference.py       # LLM inference runner — OpenAI client, structured logging
├── app.py             # FastAPI REST API server + interactive dashboard
├── validate.py        # Pre-submission validator
├── openenv.yaml       # OpenEnv specification file
├── requirements.txt   # Python dependencies
├── Dockerfile         # python:3.11-slim, port 7860
├── README.md          # This file
└── static/
    ├── index.html     # Interactive triage dashboard (vanilla HTML + CSS + JS)
    └── style.css      # Dashboard stylesheet
```

---

## 🧪 Validation

```bash
# Run all pre-submission checks
python validate.py
# Expected: ALL VALIDATION CHECKS PASSED

# Single case reward sanity check
python -c "
from environment import MedicalTriageEnv, Action
env = MedicalTriageEnv(seed=42)
obs = env.reset(case_ids=['PT-001'], shuffle=False)
action = Action(triage_level=4, recommended_actions=['wound care', 'tetanus'], critical_flags=[])
_, reward, done, info = env.step(action)
assert 0.0 <= reward.score <= 1.0
print(f'Score: {reward.score}  Breakdown: {reward.breakdown}')
print('All assertions passed.')
"
```

---

## 📦 Requirements

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
openai>=1.30.0
python-multipart>=0.0.9
aiofiles>=23.2.0
```

---

## ⚙️ Resource Requirements

| Metric | Value |
|---|---|
| LLM calls per full run | 15 (3 tasks × 5 cases) |
| Total runtime (gpt-4o-mini) | ~65 seconds |
| Maximum allowed runtime | 20 minutes |
| Hardware requirement | 2 vCPU / 8 GB RAM (no GPU needed) |
| Docker base image | `python:3.11-slim` |

---

## 🏆 Pre-Submission Checklist

| Check | Status |
|---|---|
| `inference.py` in project root | ✅ |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` read from environment | ✅ |
| All LLM calls via `openai.OpenAI` client | ✅ |
| `[START]` / `[STEP]` / `[END]` structured logs with correct field names | ✅ |
| `step=` is 0-indexed integer | ✅ |
| 3 graded tasks with scores in [0.0, 1.0] | ✅ |
| `openenv.yaml` with `observation_space`, `action_space`, `reward`, `docker` | ✅ |
| `POST /reset` returns typed `Observation` | ✅ |
| `POST /step` returns `(Observation, Reward, done, info)` | ✅ |
| `GET /state` returns full environment state | ✅ |
| `GET /health` returns HTTP 200 | ✅ |
| Dockerfile builds, exposes port 7860 | ✅ |
| Runtime < 20 min on 2 vCPU / 8 GB | ✅ (~65s) |
| `validate.py` passes all checks | ✅ |
| Grader returns variable scores (not fixed) | ✅ (5 distinct values across 25 cases) |
| Score discriminates quality (correct vs wrong action gap > 0.3) | ✅ (gap = 0.65) |

---

## 🙏 Acknowledgements

- **ESI Triage System** — Agency for Healthcare Research and Quality (AHRQ)
- **Clinical Case Design** — Emergency medicine textbooks and validated scoring systems: TIMI, Alvarado, Wells PE, Ottawa Rules, CRB-65, GCS
- **OpenEnv Framework** — Meta & Hugging Face hackathon environment standardisation
- **OpenRouter** — Multi-provider OpenAI-compatible inference API

---

*MIT License — see [LICENSE](LICENSE) for details.*
