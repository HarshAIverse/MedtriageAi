"""
inference.py - OpenEnv Inference Runner
Medical Triage & Clinical Decision Support

Reads environment variables:
    API_BASE_URL   - OpenAI-compatible API base URL
    MODEL_NAME     - Model identifier (e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct")
    HF_TOKEN       - HuggingFace / provider API token

Runs all 3 tasks sequentially, emitting structured stdout logs per OpenEnv spec:
    [START] <task_id>
    [STEP]  <task_id> <case_index> <patient_id> <score>
    [END]   <task_id> <final_score>

Total runtime target: < 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from openai import OpenAI

from environment import Action, MedicalTriageEnv, Observation
from tasks import TASK_REGISTRY, grade_task_1, grade_task_2, grade_task_3, run_episode


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

MAX_RETRIES: int = 3
RETRY_DELAY_S: float = 2.0
REQUEST_TIMEOUT_S: float = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────────────────────────────────

def build_client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError(
            "No API token found. Set HF_TOKEN or OPENAI_API_KEY environment variable."
        )
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert emergency medicine physician and triage nurse with board certification in Emergency Medicine and 15 years of ICU and ED experience. You are an ACCURATE triager — not a conservatively biased one.

You will be given a patient case with vital signs, symptoms, and medical history. Your task is to:
1. Assign an ESI (Emergency Severity Index) triage priority level (1-5)
2. Recommend specific, actionable clinical interventions
3. Identify any life-threatening conditions that must be flagged immediately

ESI LEVELS — CALIBRATED DEFINITIONS:
- 1 = IMMEDIATE: Patient requires IMMEDIATE life-saving intervention RIGHT NOW.
  Examples: unresponsive/GCS<8, active cardiac arrest, major hemorrhage, complete airway obstruction, status epilepticus, BP<70 systolic
- 2 = EMERGENT: Patient is in a high-risk situation that should NOT wait. Vital signs may be abnormal.
  Examples: active chest pain (ACS), suspected stroke within window, sepsis/urosepsis, ectopic pregnancy with bleeding, thunderclap headache (SAH rule-out), meningitis signs
- 3 = URGENT: Patient is STABLE but needs multiple diagnostic resources (labs + imaging or specialist consult).
  Examples: stable abdominal pain needing CT+labs, pneumonia needing CXR+antibiotics, possible fracture needing X-ray+ortho, stable renal colic
- 4 = LESS URGENT: Patient is STABLE and needs only ONE resource (one test or one treatment).
  Examples: minor laceration needing suture, ankle sprain needing X-ray, strep throat needing swab, minor UTI needing dipstick
- 5 = NON-URGENT: Patient is STABLE and needs NO resources — just reassurance or prescription.
  Examples: mild cold/URTI (no test needed), medication refill only, chronic back pain with no red flags, minor complaint

CRITICAL CALIBRATION RULES:
1. NEVER assign ESI 1 or 2 to a patient who is fully alert (GCS 15), has stable vitals (BP>100, HR<110, SpO2>95, RR<20), and has a benign presentation. That is WRONG.
2. ESI 5 is appropriate for: viral URTI, minor scrapes, medication refills, chronic stable conditions flaring. Do NOT upgrade these to ESI 3.
3. ESI 4 is appropriate for: minor wounds needing 1 treatment, clear UTI, ankle sprain needing X-ray only, strep throat.
4. A patient who needs BOTH a blood test AND imaging (or specialist) = ESI 3 minimum.
5. ESI 2 requires either: abnormal vitals, high-risk chief complaint (ACS/stroke/sepsis/ectopic), or severe pain in a dangerous context.
6. ESI 1 requires: needing a life-saving intervention in the NEXT MINUTES (intubation, CPR, emergent surgery, defibrillation).
7. In critical_flags, be specific: say "acute ischemic stroke", "septic shock", "aortic dissection" — not vague terms.
8. Do NOT over-triage. Over-triaging stable patients (giving ESI 1-2 to ESI 4-5 patients) is explicitly penalized.

You MUST respond with a valid JSON object matching this exact schema:
{
  "triage_level": <integer 1-5>,
  "recommended_actions": [<string>, ...],
  "critical_flags": [<string>, ...],
  "reasoning": "<string>"
}
No markdown, no explanation outside the JSON object."""



def build_user_prompt(obs: Observation) -> str:
    v = obs.vitals
    h = obs.history
    return f"""PATIENT CASE {obs.patient_id}  (Case {obs.case_index + 1} of {obs.total_cases})
Presentation time: {obs.timestamp}

=== VITAL SIGNS ===
Heart Rate:           {v.heart_rate} bpm
Blood Pressure:       {v.systolic_bp}/{v.diastolic_bp} mmHg
Respiratory Rate:     {v.respiratory_rate} breaths/min
Temperature:          {v.temperature_celsius}°C
SpO2:                 {v.oxygen_saturation}%
GCS:                  {v.glasgow_coma_scale}/15
Pain Score:           {v.pain_score}/10

=== PRESENTING SYMPTOMS ===
{chr(10).join(f"• {s}" for s in obs.symptoms)}

=== PATIENT HISTORY ===
Age/Sex:              {h.age}y {h.sex}
Chief Complaint:      {h.chief_complaint}
Symptom Onset:        {h.onset_hours:.1f} hours ago
Past Medical History: {", ".join(h.past_medical_history) or "None"}
Current Medications:  {", ".join(h.current_medications) or "None"}
Allergies:            {", ".join(h.allergies) or "NKDA"}
Surgical History:     {", ".join(h.surgical_history) or "None"}

=== ADDITIONAL CONTEXT ===
{obs.additional_context or "No additional notes."}

Provide your triage assessment as a JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(data: dict) -> Action:
    """
    Robustly parse LLM JSON output into an Action.
    Handles: string triage_level, null fields, non-list recommended_actions,
    out-of-range ESI values, and missing keys.
    """
    # --- triage_level: coerce string/float/None -> int in [1,5] ---
    raw_level = data.get("triage_level", 3)
    try:
        level = max(1, min(5, int(float(str(raw_level).strip()))))
    except (ValueError, TypeError):
        level = 3  # safe default

    # --- recommended_actions: must be list[str] ---
    raw_actions = data.get("recommended_actions") or data.get("actions") or []
    if isinstance(raw_actions, str):
        raw_actions = [a.strip() for a in raw_actions.split(",") if a.strip()]
    elif not isinstance(raw_actions, list):
        raw_actions = ["clinical assessment"]
    actions = [str(a) for a in raw_actions if a] or ["clinical assessment"]

    # --- critical_flags: must be list[str] ---
    raw_flags = data.get("critical_flags") or data.get("flags") or []
    if isinstance(raw_flags, str):
        raw_flags = [f.strip() for f in raw_flags.split(",") if f.strip()]
    elif not isinstance(raw_flags, list):
        raw_flags = []
    flags = [str(f) for f in raw_flags if f]

    # --- reasoning: optional string ---
    reasoning = str(data.get("reasoning") or data.get("rationale") or "")

    return Action(
        triage_level=level,
        recommended_actions=actions,
        critical_flags=flags,
        reasoning=reasoning,
    )


class LLMTriageAgent:
    """Calls the LLM to produce an Action for each patient Observation."""


    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self._call_count = 0
        self._total_latency = 0.0

    def __call__(self, obs: Observation) -> Action:
        user_msg = build_user_prompt(obs)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                latency = time.monotonic() - t0
                self._call_count += 1
                self._total_latency += latency

                raw_json = response.choices[0].message.content
                data = json.loads(raw_json)
                return _parse_llm_response(data)

            except json.JSONDecodeError as e:
                _log(f"  [WARN] JSON parse error on attempt {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    return _safe_fallback_action()
                time.sleep(RETRY_DELAY_S)

            except Exception as e:  # noqa: BLE001
                _log(f"  [WARN] API error on attempt {attempt}: {type(e).__name__}: {e}")
                if attempt == MAX_RETRIES:
                    return _safe_fallback_action()
                time.sleep(RETRY_DELAY_S * attempt)

        return _safe_fallback_action()

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency_s": round(self._total_latency, 2),
            "avg_latency_s": (
                round(self._total_latency / self._call_count, 2) if self._call_count else 0.0
            ),
        }


def _safe_fallback_action() -> Action:
    """Conservative ESI 3 fallback when the LLM fails."""
    return Action(
        triage_level=3,
        recommended_actions=["full clinical assessment", "vital sign monitoring", "IV access"],
        critical_flags=[],
        reasoning="Fallback: LLM parsing failed. Defaulting to ESI 3 for safety.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, flush=True)


def _emit_start(task_id: str, meta) -> None:
    _log(f"[START] task_id={task_id} name={meta.name!r} difficulty={meta.difficulty} "
         f"target={meta.target_score} cases={meta.case_ids}")


def _emit_step(task_id: str, case_index: int, patient_id: str, score: float,
               triage_level: int, correct_level: int) -> None:
    _log(
        f"[STEP]  task_id={task_id} step={case_index} patient={patient_id} "
        f"score={score:.4f} assigned_level={triage_level} correct_level={correct_level}"
    )


def _emit_end(task_id: str, final_score: float, target: float, wall_time: float) -> None:
    status = "PASS" if final_score >= target else "FAIL"
    _log(
        f"[END]   task_id={task_id} score={final_score:.4f} "
        f"target={target} status={status} wall_time_s={wall_time:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task runners with structured logging
# ─────────────────────────────────────────────────────────────────────────────

def run_task_with_logging(
    task_id: str,
    agent: LLMTriageAgent,
) -> float:
    """Run a single task, emitting [START]/[STEP]/[END] logs. Returns final grade."""
    meta = TASK_REGISTRY[task_id]
    _emit_start(task_id, meta)

    env = MedicalTriageEnv(seed=42)
    obs = env.reset(case_ids=meta.case_ids, shuffle=False)

    step_scores: list[float] = []
    all_actions: list[dict] = []
    all_infos: list[dict] = []

    t0 = time.monotonic()
    done = False

    while not done:
        action = agent(obs)
        next_obs, reward, done, info = env.step(action)

        step_scores.append(reward.score)
        all_actions.append(action.model_dump())
        all_infos.append(info)

        _emit_step(
            task_id=task_id,
            case_index=info.get("episode_scores_so_far", [0]).__len__() - 1,
            patient_id=info["case_id"],
            score=reward.score,
            triage_level=action.triage_level,
            correct_level=info["correct_triage_level"],
        )
        obs = next_obs

    wall_time = time.monotonic() - t0

    # Apply task-specific weighting / adjustments
    if task_id == "task_1":
        from tasks import grade_task_1
        # Re-use the grader but log interim
        final_grade = _compute_task1_grade(step_scores)
    elif task_id == "task_2":
        final_grade = _compute_task2_grade(step_scores, all_infos)
    else:
        final_grade = _compute_task3_grade(step_scores, all_infos, all_actions)

    _emit_end(task_id, final_grade, meta.target_score, wall_time)
    return final_grade


def _compute_task1_grade(scores: list[float]) -> float:
    return round(max(0.0, min(1.0, sum(scores) / len(scores))), 4)


def _compute_task2_grade(scores: list[float], infos: list[dict]) -> float:
    ambiguous = {"PT-009", "PT-011"}
    weighted_sum = 0.0
    total_weights = 0.0
    for score, info in zip(scores, infos):
        w = 1.5 if info["case_id"] in ambiguous else 1.0
        weighted_sum += score * w
        total_weights += w
    return round(max(0.0, min(1.0, weighted_sum / total_weights)), 4)


def _compute_task3_grade(
    scores: list[float], infos: list[dict], actions: list[dict]
) -> float:
    base = sum(scores) / len(scores) if scores else 0.0
    adj = 0.0
    for info, act in zip(infos, actions):
        gt_flags = info["correct_critical_flags"]
        pred_flags = act.get("critical_flags", [])
        if gt_flags:
            gt_tokens = set(" ".join(f.lower() for f in gt_flags).split())
            pred_tokens = set(" ".join(f.lower() for f in pred_flags).split())
            recall = len(gt_tokens & pred_tokens) / max(len(gt_tokens), 1)
            if recall == 1.0:
                adj += 0.05
            elif recall == 0.0:
                adj -= 0.05
    return round(max(0.0, min(1.0, base + adj)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _log("=" * 70)
    _log("Medical Triage Environment - OpenEnv Inference Runner")
    _log("=" * 70)
    _log(f"API Base URL : {API_BASE_URL}")
    _log(f"Model        : {MODEL_NAME}")
    _log(f"Token set    : {'YES' if HF_TOKEN else 'NO ← set HF_TOKEN or OPENAI_API_KEY'}")
    _log("")

    client = build_client()
    agent = LLMTriageAgent(client=client, model=MODEL_NAME)

    task_results: dict[str, float] = {}
    total_t0 = time.monotonic()

    for task_id in ["task_1", "task_2", "task_3"]:
        _log("")
        score = run_task_with_logging(task_id, agent)
        task_results[task_id] = score

    total_wall = time.monotonic() - total_t0

    _log("")
    _log("=" * 70)
    _log("FINAL RESULTS")
    _log("=" * 70)
    for task_id, score in task_results.items():
        meta = TASK_REGISTRY[task_id]
        status = "[OK] PASS" if score >= meta.target_score else "[FAIL] FAIL"
        _log(
            f"  {task_id}  [{meta.difficulty:6s}]  score={score:.4f}  "
            f"target>={meta.target_score}  {status}"
        )

    overall = sum(task_results.values()) / len(task_results)
    _log(f"\n  Overall mean score : {overall:.4f}")
    _log(f"  Total wall time    : {total_wall:.1f}s")
    _log(f"  LLM call stats     : {agent.stats}")
    _log("=" * 70)

    # Exit with non-zero if any task fails (useful for CI)
    all_passed = all(
        task_results[tid] >= TASK_REGISTRY[tid].target_score for tid in task_results
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
