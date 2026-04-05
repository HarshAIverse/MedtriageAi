# 🏥 Medical Triage AI — OpenEnv Submission

> **Hackathon Submission** | Author: [HarshAIverse](https://huggingface.co/HarshAIverse) | Version: 1.0.0 | License: MIT

An RL environment where an AI agent acts as an **emergency department triage nurse**, processing incoming patient cases and making structured, evidence-based clinical decisions under time pressure.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://huggingface.co/spaces/HarshAIverse/medical-triage-env
cd medical-triage-env

# 2. Install
pip install -r requirements.txt

# 3. Set environment variables (required)
export API_BASE_URL=https://openrouter.ai/api/v1   # or https://api.openai.com/v1
export MODEL_NAME=openai/gpt-4o-mini               # or gpt-4o, gpt-4-turbo, etc.
export HF_TOKEN=sk-or-v1-...                       # your API key

# 4. Run inference (produces [START]/[STEP]/[END] structured logs)
python inference.py

# 5. Start the interactive dashboard
uvicorn app:app --host 0.0.0.0 --port 7860
# → Open http://localhost:7860
```

---

## ✅ Pre-Submission Checklist

All items below have been verified. Every automated check passes.

| # | Check | Status |
|---|---|---|
| 1 | `inference.py` in project root | ✅ |
| 2 | `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars read by inference | ✅ |
| 3 | Uses `openai.OpenAI` client for all LLM calls | ✅ |
| 4 | Structured `[START]`/`[STEP]`/`[END]` stdout logs emitted | ✅ |
| 5 | 3 graded tasks with scores verified in `[0.0, 1.0]` | ✅ |
| 6 | `openenv.yaml` spec file with all required fields | ✅ |
| 7 | `POST /reset` endpoint returns typed `Observation` | ✅ |
| 8 | `POST /step` endpoint returns `(Observation, Reward, done, info)` | ✅ |
| 9 | `GET /state` endpoint returns full environment state | ✅ |
| 10 | `Dockerfile` builds on Python 3.11-slim, exposes port 7860 | ✅ |
| 11 | Runtime < 20 min on 2 vCPU / 8 GB RAM | ✅ (~65 sec for 15 calls) |
| 12 | All reward scores clamped to `[0.0, 1.0]` | ✅ |
| 13 | HF Space serves `GET /` returning HTTP 200 | ✅ |
| 14 | `validate.py` passes all checks | ✅ |

Run the built-in validator yourself:

```bash
python validate.py
# Expected output: ALL VALIDATION CHECKS PASSED
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ Yes | OpenAI-compatible API base URL (e.g. `https://openrouter.ai/api/v1`) |
| `MODEL_NAME` | ✅ Yes | Model identifier (e.g. `openai/gpt-4o-mini`, `gpt-4o`) |
| `HF_TOKEN` | ✅ Yes | API key / HuggingFace token for the inference provider |

> **Windows PowerShell:**
> ```powershell
> $env:API_BASE_URL = "https://openrouter.ai/api/v1"
> $env:MODEL_NAME   = "openai/gpt-4o-mini"
> $env:HF_TOKEN     = "sk-or-v1-..."
> python inference.py
> ```
>
> **Linux / macOS / Docker:**
> ```bash
> export API_BASE_URL="https://openrouter.ai/api/v1"
> export MODEL_NAME="openai/gpt-4o-mini"
> export HF_TOKEN="sk-or-v1-..."
> python inference.py
> ```

---

## 📋 Structured Log Format

The inference script emits exactly this format to stdout (required by the OpenEnv evaluator):

```
[START] task_id=task_1 name='Stable Patient Triage' difficulty=easy target=0.6 cases=['PT-001', ...]
[STEP]  task_id=task_1 step=0 patient=PT-001 score=0.7348 assigned_level=4 correct_level=4
[STEP]  task_id=task_1 step=1 patient=PT-002 score=0.7167 assigned_level=4 correct_level=4
[STEP]  task_id=task_1 step=2 patient=PT-003 score=0.4962 assigned_level=4 correct_level=5
[STEP]  task_id=task_1 step=3 patient=PT-004 score=0.4667 assigned_level=4 correct_level=5
[STEP]  task_id=task_1 step=4 patient=PT-005 score=0.7000 assigned_level=5 correct_level=5
[END]   task_id=task_1 score=0.6264 target=0.6 status=PASS wall_time_s=13.08

[START] task_id=task_2 ...
...
[END]   task_id=task_2 score=0.4432 target=0.4 status=PASS wall_time_s=21.61

[START] task_id=task_3 ...
...
[END]   task_id=task_3 score=0.2066 target=0.2 status=PASS wall_time_s=25.95
```

**Field definitions (do not deviate):**

| Token | Field | Type | Example |
|---|---|---|---|
| `[START]` | `task_id` | string | `task_1` |
| `[START]` | `name` | quoted string | `'Stable Patient Triage'` |
| `[START]` | `difficulty` | string | `easy` |
| `[START]` | `target` | float | `0.6` |
| `[START]` | `cases` | list | `['PT-001',...]` |
| `[STEP]` | `task_id` | string | `task_1` |
| `[STEP]` | `step` | int (0-indexed) | `0` |
| `[STEP]` | `patient` | string | `PT-001` |
| `[STEP]` | `score` | float (4dp) | `0.7348` |
| `[STEP]` | `assigned_level` | int 1–5 | `4` |
| `[STEP]` | `correct_level` | int 1–5 | `4` |
| `[END]` | `task_id` | string | `task_1` |
| `[END]` | `score` | float (4dp) | `0.6264` |
| `[END]` | `target` | float | `0.6` |
| `[END]` | `status` | `PASS` or `FAIL` | `PASS` |
| `[END]` | `wall_time_s` | float (2dp) | `13.08` |

---

## 📐 Tasks

### Task 1 — Stable Patient Triage (Easy)

> **Target score: ≥ 0.60** | Cases: PT-001, PT-002, PT-003, PT-004, PT-005

5 clearly stable patients with benign presentations: laceration, ankle sprain, strep pharyngitis, mechanical back pain, viral URTI. All vitals are normal. The agent must correctly assign ESI 4–5.

**Key pitfall:** Over-triaging stable patients (ESI 1–2 when correct is ESI 4–5) incurs a −0.10 penalty per case.

---

### Task 2 — Mixed Urgency Triage (Medium)

> **Target score: ≥ 0.40** | Cases: PT-006, PT-008, PT-009, PT-010, PT-011

5 mixed-acuity cases including 2 with ambiguous presentations (weighted 1.5×):
- PT-006: Classic ACS — ESI 2
- PT-008: Renal colic with morphine allergy — ESI 3
- PT-009: Urosepsis in elderly diabetic *(ambiguous, 1.5× weight)* — ESI 2
- PT-010: Possible appendicitis (Alvarado 7) — ESI 3
- PT-011: Thunderclap headache / SAH *(ambiguous, 1.5× weight)* — ESI 2

---

### Task 3 — Complex Multi-System Critical Triage (Hard)

> **Target score: ≥ 0.20** | Cases: PT-012, PT-014, PT-015, PT-019, PT-024

5 cases designed to fool pattern-matching agents with deceptive presentations:

| Case | Condition | Hidden trap |
|---|---|---|
| PT-012 | Opioid overdose | Partial naloxone response — airway still at risk |
| PT-014 | Acute ischemic stroke | On apixaban — complicates tPA protocol |
| PT-015 | 3rd-degree heart block | Patient calls it "anxiety" — HR 38 |
| PT-019 | Meningococcal septicemia | Penicillin anaphylaxis — ceftriaxone is the answer |
| PT-024 | Digoxin toxicity | Appears well and mobile — HR 44, nausea dismissed |

Critical flag miss penalty: −0.05 per case where critical flags are entirely absent.

---

## 🏆 Baseline Scores

| Task | Naive Heuristic | GPT-4o-mini | GPT-4o | Target |
|---|---|---|---|---|
| Task 1 (Easy) | ~0.19 | ~0.63 | ~0.88 | **≥ 0.60** |
| Task 2 (Medium) | ~0.08 | ~0.44 | ~0.76 | **≥ 0.40** |
| Task 3 (Hard) | ~0.00 | ~0.21 | ~0.62 | **≥ 0.20** |
| **Overall** | **~0.09** | **~0.43** | **~0.75** | — |

> The naive heuristic agent uses only SpO2/HR/BP/GCS thresholds — no clinical reasoning.
> GPT-4o-mini scores are measured at temperature=0, 15 API calls total.

---

## 📊 Reward Function

```
score = (
    0.50 × triage_level_accuracy    # 1.0 exact, 0.5 for ±1 level, 0.0 for ≥2 levels off
  + 0.20 × action_concept_coverage  # synonym+bigram aware (not strict keyword match)
  + 0.20 × critical_flag_recall     # proportion of ground-truth critical flags identified
  + 0.10 × flag_bonus               # bonus for correctly catching critical flags
  - 0.30 × missing_flag_penalty     # per-flag penalty for entirely missing a critical flag
  - 0.10 × overtriage_penalty       # penalty for ESI 1–2 on ESI 4–5 patients
)
# Clamped to [0.0, 1.0]
```

**Key design decisions:**
- **Concept-aware action scorer**: Uses synonym groups and bigram matching to handle LLM paraphrasing (e.g. "oxygen supplementation" matches "O2 therapy")
- **Robust JSON coercion**: String triage levels (`"4"`), null fields, and comma-separated lists are all handled gracefully
- **Over-triage penalty**: Explicitly penalises assigning ESI 1–2 to clearly stable (ESI 4–5) patients

---

## 🔌 REST API

The environment runs as a FastAPI server. The dashboard is served at `GET /`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive dashboard (HTML) |
| `GET` | `/health` | Health check — returns `{"status":"ok"}` |
| `POST` | `/reset` | Start episode. Body: `{"case_ids": [...], "shuffle": false}` |
| `POST` | `/step` | Submit action. Body: `{"action": {triage_level, recommended_actions, critical_flags}}` |
| `GET` | `/state` | Full environment state |
| `GET` | `/cases` | All 25 case IDs grouped by difficulty |
| `GET` | `/case/{patient_id}` | Metadata for a specific case |
| `GET` | `/tasks` | Task definitions with targets |
| `POST` | `/run_task` | Run full task with heuristic baseline agent |
| `GET` | `/docs` | Swagger interactive API docs |

### Example: start an episode and submit a triage action

```bash
# Start Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"case_ids": ["PT-001","PT-002","PT-003","PT-004","PT-005"], "shuffle": false}'

# Submit triage for PT-001 (minor cut)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "triage_level": 4,
      "recommended_actions": ["wound irrigation", "suture closure", "tetanus assessment"],
      "critical_flags": [],
      "reasoning": "Minor laceration, normal vitals, no systemic involvement."
    }
  }'
```

### Response from `/step`

```json
{
  "observation": { "patient_id": "PT-002", "vitals": {...}, "symptoms": [...], ... },
  "reward": {
    "score": 0.7348,
    "breakdown": {
      "triage_level_score": 1.0,
      "action_coverage_score": 0.45,
      "flag_recall_score": 1.0,
      "critical_flag_bonus": 0.0,
      "missing_flag_penalty": -0.0,
      "overtriage_penalty": -0.0,
      "total": 0.7348
    },
    "feedback": "Correct ESI level. Good action coverage."
  },
  "done": false,
  "info": {
    "case_id": "PT-001",
    "correct_triage_level": 4,
    "correct_critical_flags": [],
    "difficulty": "easy"
  }
}
```

---

## 🗂️ Observation Space

Each `Observation` is a Pydantic model returned by `/reset` and `/step`:

| Field | Type | Description |
|---|---|---|
| `patient_id` | `str` | Unique identifier (e.g. `PT-012`) |
| `vitals.heart_rate` | `int` (bpm) | Heart rate |
| `vitals.systolic_bp` | `int` (mmHg) | Systolic blood pressure |
| `vitals.diastolic_bp` | `int` (mmHg) | Diastolic blood pressure |
| `vitals.respiratory_rate` | `int` (br/min) | Respiratory rate |
| `vitals.temperature_celsius` | `float` (°C) | Core temperature |
| `vitals.oxygen_saturation` | `float` (%) | SpO2 |
| `vitals.glasgow_coma_scale` | `int` [3–15] | GCS total |
| `vitals.pain_score` | `int` [0–10] | Numeric pain scale |
| `symptoms` | `list[str]` | Presenting symptoms |
| `history.age` / `history.sex` | `int` / `str` | Demographics |
| `history.chief_complaint` | `str` | Primary ED presentation |
| `history.onset_hours` | `float` | Hours since onset |
| `history.past_medical_history` | `list[str]` | Relevant diagnoses |
| `history.current_medications` | `list[str]` | Active medications |
| `history.allergies` | `list[str]` | Drug allergies |
| `history.surgical_history` | `list[str]` | Past surgeries |
| `additional_context` | `str` | Nurse notes, labs, clinical scores |
| `timestamp` | `str` (ISO-8601) | Presentation time |
| `case_index` | `int` | 0-based position in episode |
| `total_cases` | `int` | Episode length |

---

## ⚙️ Action Space

Each `Action` is a Pydantic model submitted to `/step`:

| Field | Type | Constraint | Description |
|---|---|---|---|
| `triage_level` | `int` | 1–5 | ESI priority level |
| `recommended_actions` | `list[str]` | min 1 item | Ordered clinical interventions |
| `critical_flags` | `list[str]` | can be empty | Life-threatening suspected diagnoses |
| `reasoning` | `str` | optional | Clinical reasoning / differential |

**ESI Level Reference:**

| Level | Label | Definition | Examples |
|---|---|---|---|
| 1 | Immediate | Life-saving intervention required NOW | Cardiac arrest, airway obstruction, GCS <8 |
| 2 | Emergent | High-risk, cannot wait | ACS, stroke, sepsis, ectopic pregnancy |
| 3 | Urgent | Stable, needs multiple resources | Pneumonia, appendicitis, renal colic |
| 4 | Less Urgent | Stable, needs 1 resource | Sprain + X-ray, minor laceration, strep swab |
| 5 | Non-Urgent | Minor, no resources needed | URTI, medication refill, benign complaint |

---

## 🐳 Docker

```bash
# Build
docker build -t medical-triage-env:latest .

# Run (with env variables)
docker run -d \
  --name triage-env \
  -p 7860:7860 \
  -e API_BASE_URL="https://openrouter.ai/api/v1" \
  -e MODEL_NAME="openai/gpt-4o-mini" \
  -e HF_TOKEN="sk-or-v1-..." \
  medical-triage-env:latest

# Health check
curl http://localhost:7860/health

# Run inference inside container
docker exec triage-env python inference.py
```

---

## 📁 File Structure

```
medical-triage-env/
├── environment.py     # Core RL env: Pydantic models, 25 patient cases, MedicalTriageEnv
├── tasks.py           # Task definitions, TASK_REGISTRY, graders (grade_task_1/2/3)
├── inference.py       # LLM inference runner — OpenAI client, [START]/[STEP]/[END] logs
├── app.py             # FastAPI REST API server + static dashboard hosting
├── openenv.yaml       # OpenEnv specification (observation/action/reward/docker)
├── validate.py        # Pre-submission validator — run before submitting
├── requirements.txt   # Python dependencies (fastapi, openai, pydantic, uvicorn)
├── Dockerfile         # Python 3.11-slim, port 7860, includes static/
├── README.md          # This file
└── static/
    ├── index.html     # Interactive triage dashboard (vanilla HTML+CSS+JS)
    └── style.css      # Complete dashboard stylesheet
```

---

## 📦 Dependencies

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
openai>=1.30.0
python-multipart>=0.0.9
aiofiles>=23.2.0
```

Install: `pip install -r requirements.txt`

---

## 🧪 Validation

```bash
# Full pre-submission check (run this before submitting)
python validate.py

# Quick reward function sanity check
python -c "
from environment import MedicalTriageEnv, Action
env = MedicalTriageEnv(seed=42)
obs = env.reset(case_ids=['PT-001'], shuffle=False)
action = Action(triage_level=4, recommended_actions=['wound care', 'tetanus'], critical_flags=[])
obs2, reward, done, info = env.step(action)
assert 0.0 <= reward.score <= 1.0, 'Score out of range!'
print(f'Score: {reward.score}  Breakdown: {reward.breakdown}')
print('OK — all assertions passed.')
"

# Run tasks.py baseline grader
python tasks.py
```

---

## 📊 Resource Requirements

| Metric | Measured |
|---|---|
| Total inference runtime (gpt-4o-mini, 15 calls) | ~65 seconds |
| Avg latency per LLM call | ~4 seconds |
| Max runtime allowed | 20 minutes |
| Hardware target | 2 vCPU / 8 GB RAM |
| Docker image base | python:3.11-slim (~300 MB) |
| LLM calls per full run | 15 (3 tasks × 5 cases) |

---

## 🌐 HuggingFace Space

Live URL: **https://huggingface.co/spaces/HarshAIverse/medical-triage-env**

The Space serves:
- `GET /` → Interactive triage dashboard
- `GET /health` → `{"status": "ok"}` (200)
- All REST API endpoints documented above

---

## 🙏 Acknowledgements

- **ESI Triage System** — Agency for Healthcare Research and Quality (AHRQ)
- **Clinical Cases** — Informed by emergency medicine textbooks and validated scoring systems (TIMI, Alvarado, Wells PE, Ottawa Rules, CRB-65, GCS)
- **OpenEnv Framework** — Hackathon environment standardization
- **OpenRouter** — Multi-provider LLM inference API

---

*MIT License — see LICENSE for details.*
