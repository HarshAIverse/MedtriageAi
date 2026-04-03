# 🏥 Medical Triage & Clinical Decision Support Environment

> **OpenEnv Hackathon Submission** | [HarshAIverse](https://huggingface.co/HarshAIverse) | v1.0.0

An RL environment where an AI agent acts as an **emergency department triage nurse**, processing incoming patient cases and making structured, evidence-based clinical decisions under time pressure.

---

## 🎯 Real-World Motivation

Emergency Department triage errors cause preventable deaths and waste limited resources. The ESI (Emergency Severity Index) triage system is the gold standard, but even trained nurses make errors — particularly when patients present with atypical symptoms or misleading vitals (e.g., elderly patients who maintain normal BP despite haemorrhage, or anxiety disguising complete heart block).

This environment trains AI agents to:
1. **Correctly prioritise** patients using the 5-level ESI system
2. **Recommend evidence-based interventions** matching clinical guidelines
3. **Flag life-threatening conditions** proactively — especially when hidden behind deceptive presentations

The agent faces 25 medically accurate synthetic cases ranging from trivial lacerations to aortic dissections and meningococcal septicemia, with partial-credit rewards that incentivise clinical vigilance without over-triaging stable patients.

---

## 📐 Observation Space

Each observation represents a full patient snapshot at the time of triage presentation.

| Field | Type | Description |
|---|---|---|
| `patient_id` | `string` | Unique case identifier (e.g. `PT-012`) |
| `vitals.heart_rate` | `int` (bpm) | Heart rate |
| `vitals.systolic_bp` | `int` (mmHg) | Systolic blood pressure |
| `vitals.diastolic_bp` | `int` (mmHg) | Diastolic blood pressure |
| `vitals.respiratory_rate` | `int` (breaths/min) | Respiratory rate |
| `vitals.temperature_celsius` | `float` (°C) | Core body temperature |
| `vitals.oxygen_saturation` | `float` (%) | SpO2 peripheral oxygen saturation |
| `vitals.glasgow_coma_scale` | `int` \[3–15\] | GCS total score |
| `vitals.pain_score` | `int` \[0–10\] | Numeric pain rating |
| `symptoms` | `list[str]` | Presenting symptoms as reported |
| `history.age` | `int` (years) | Patient age |
| `history.sex` | `str` \[M/F\] | Biological sex |
| `history.chief_complaint` | `str` | Primary ED presentation reason |
| `history.onset_hours` | `float` | Hours since symptom onset |
| `history.past_medical_history` | `list[str]` | Relevant diagnoses |
| `history.current_medications` | `list[str]` | Active medications |
| `history.allergies` | `list[str]` | Known drug/substance allergies |
| `history.surgical_history` | `list[str]` | Past surgical procedures |
| `additional_context` | `str` | Nurse notes, EMS report, lab/imaging results, validated clinical scores (TIMI, Alvarado, Wells PE, etc.) |
| `timestamp` | `str` (ISO-8601) | Presentation time |
| `case_index` | `int` | Zero-based position in episode |
| `total_cases` | `int` | Episode length |

---

## ⚡ Action Space

| Field | Type | Description |
|---|---|---|
| `triage_level` | `int` \[1–5\] | ESI priority: **1**=Immediate, **2**=Emergent, **3**=Urgent, **4**=Less urgent, **5**=Non-urgent |
| `recommended_actions` | `list[str]` | Ordered clinical interventions (e.g. `"IV access x2"`, `"12-lead ECG stat"`) |
| `critical_flags` | `list[str]` | Life-threatening diagnoses suspected (empty if benign) |
| `reasoning` | `str` | Optional clinical reasoning / differential |

### ESI Level Reference

| Level | Label | When to Use | Example |
|---|---|---|---|
| 1 | Immediate | Requires immediate physician resuscitation | Cardiac arrest, respiratory failure, major haemorrhage |
| 2 | Emergent | High-risk, should not wait | ACS, stroke, sepsis, ectopic pregnancy |
| 3 | Urgent | Stable, needs multiple resources | Pneumonia, appendicitis, renal colic |
| 4 | Less Urgent | Stable, needs 1 resource | Sprain + X-ray, minor laceration, UTI |
| 5 | Non-Urgent | Minor, 0 resources needed | URTI, medication refill, minor complaint |

---

## 📋 Tasks

### Task 1 — Stable Patient Triage (Easy)

| Property | Value |
|---|---|
| Difficulty | 🟢 Easy |
| Cases | 5 clearly stable patients |
| Target Score | >= 0.60 |
| Grading | Mean episode reward, clamped to [0, 1] |

Patients present with minor injuries and benign conditions: small laceration, ankle sprain (Ottawa positive), strep pharyngitis, mechanical back pain, viral URTI. All vitals are normal. The agent must assign ESI 4–5 and recommend appropriate outpatient actions.

**Key pitfall**: Over-triaging stable patients (assigning ESI 1–2) incurs a -0.1 penalty per case.

---

### Task 2 — Mixed Urgency Triage (Medium)

| Property | Value |
|---|---|
| Difficulty | 🟡 Medium |
| Cases | 5 mixed cases — 2 with ambiguous symptoms |
| Target Score | >= 0.40 |
| Grading | Weighted mean (ambiguous cases worth 1.5×) |

Cases include: classic ACS presentation (ESI 2), renal colic with morphine allergy (ESI 3), urosepsis in elderly diabetic (ESI 2), high-probability appendicitis (ESI 3), and thunderclap headache requiring SAH exclusion (ESI 2).

**Key pitfall**: Urosepsis (PT-009) presents with deceptively moderate vitals but GCS drop makes it ESI 2. Thunderclap headache (PT-011) must not be dismissed as migraine.

---

### Task 3 — Complex Multi-System Critical Triage (Hard)

| Property | Value |
|---|---|
| Difficulty | 🔴 Hard |
| Cases | 5 critical cases with misleading features |
| Target Score | >= 0.20 |
| Grading | Mean reward ± flag recall bonuses/penalties |

Cases designed to fool pattern-matching agents:

| Case | Condition | Hidden trap |
|---|---|---|
| PT-012 | Opioid overdose | Partial naloxone response — airway still at risk |
| PT-014 | Acute ischemic stroke | On apixaban — tPA protocol complicated |
| PT-015 | 3rd-degree heart block | Patient attributes symptoms to "anxiety" |
| PT-019 | Meningococcal septicemia | Penicillin anaphylaxis — ceftriaxone choice critical |
| PT-024 | Digoxin toxicity | Patient appears well and mobile — deceptively low concern |

**Critical flag miss penalty**: -0.05 per case where critical flags are entirely missed.

---

## 🏆 Expected / Baseline Scores

| Task | Naive Vitals Heuristic | GPT-4o-mini | GPT-4o | Target |
|---|---|---|---|---|
| Task 1 (Easy) | ~0.19 | ~0.63 | ~0.88 | >= 0.60 |
| Task 2 (Medium) | ~0.13 | ~0.44 | ~0.76 | >= 0.40 |
| Task 3 (Hard) | ~0.00 | ~0.21 | ~0.62 | >= 0.20 |
| **Overall** | **~0.11** | **~0.43** | **~0.75** | — |

> Baseline heuristic triages using SpO2/HR/BP/GCS thresholds only, with no clinical reasoning.

---

## 🔧 Reward Function

```
total_score = (
    0.50 x triage_level_accuracy    # 1.0 exact, 0.5 for +-1, 0.0 for >=2 off
  + 0.20 x action_concept_coverage  # synonym+bigram aware, not strict word-match
  + 0.20 x critical_flag_recall     # proportion of critical flags identified
  + up to 0.10 x flag_bonus         # bonus for catching flags (proportional to recall)
  - up to 0.30 x missing_flag_frac  # penalty for missing life-threatening flags
  - 0.10 x overtriage_indicator     # penalty for ESI 1-2 on ESI 4-5 patients
)
```
All scores clamped to **[0.0, 1.0]**.

---

## 🚀 Setup & Running

### Prerequisites

- Docker 20.10+
- Python 3.11+ (for local development)
- An OpenAI-compatible API key (OpenAI, Together AI, HF Inference, etc.)

---

### Option A — Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/HarshAIverse/medical-triage-env
cd medical-triage-env

# 2. Build the image
docker build -t medical-triage-env:latest .

# 3. Run the API server
docker run -d \
  --name triage-env \
  -p 7860:7860 \
  -e HF_TOKEN=your_token_here \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  medical-triage-env:latest

# 4. Verify health check
curl http://localhost:7860/

# 5. Run inference (inside the container)
docker exec triage-env python inference.py
```

---

### Option B — Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# 3. Run inference (separate terminal)
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py
```

---

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start new episode (optionally specify `case_ids`, `shuffle`, `seed`) |
| `POST` | `/step` | Submit action, receive reward + next observation |
| `GET` | `/state` | Full environment state |
| `GET` | `/cases` | List all case IDs by difficulty |
| `GET` | `/case/{patient_id}` | Metadata for a specific case |
| `GET` | `/docs` | Interactive Swagger UI |

#### Example: Start an episode and take an action

```bash
# Reset with Task 1 cases
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"case_ids": ["PT-001","PT-002","PT-003","PT-004","PT-005"], "shuffle": false}'

# Submit a triage action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "triage_level": 4,
      "recommended_actions": ["wound cleaning", "suture closure", "tetanus assessment"],
      "critical_flags": [],
      "reasoning": "Minor laceration, all vitals normal, no systemic involvement."
    }
  }'
```

---

## 📁 File Structure

```
medical-triage-env/
├── environment.py    # Core env: Pydantic models, 25 patient cases, MedicalTriageEnv
├── tasks.py          # Task definitions and deterministic graders
├── inference.py      # LLM inference runner with structured logging
├── app.py            # FastAPI REST API wrapper
├── openenv.yaml      # OpenEnv specification file
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker image definition
└── README.md         # This file
```

---

## 🧪 Validation

```bash
# OpenEnv CLI validation
openenv validate

# Smoke test with baseline agent
python tasks.py

# Unit test reward function consistency
python -c "
from environment import MedicalTriageEnv, Action
env = MedicalTriageEnv(seed=42)
obs = env.reset(case_ids=['PT-001'], shuffle=False)
action = Action(triage_level=4, recommended_actions=['wound care', 'tetanus'], critical_flags=[])
obs2, reward, done, info = env.step(action)
assert 0.0 <= reward.score <= 1.0
print(f'Reward: {reward.score}  Breakdown: {reward.breakdown}')
print('All assertions passed.')
"
```

---

## 📊 Resource Requirements

| Metric | Target |
|---|---|
| Total inference runtime | < 20 minutes |
| Hardware target | 2 vCPU / 8 GB RAM |
| Docker image size | ~300 MB |
| LLM calls per full run | 15 (3 tasks × 5 cases) |
| Avg latency per call | ~3–8s (model dependent) |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **ESI Triage System** — Agency for Healthcare Research and Quality (AHRQ)
- **Clinical case design** — Informed by emergency medicine textbooks and validated clinical scoring systems (TIMI, Alvarado, Wells PE, CRB-65, Ottawa Rules)
- **OpenEnv Framework** — Hackathon environment standardization
