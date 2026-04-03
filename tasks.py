"""
OpenEnv Task Definitions - Medical Triage & Clinical Decision Support

Three tasks of increasing difficulty. Each task:
  1. Builds a fixed episode by selecting specific patient cases
  2. Runs the episode using the provided agent callable
  3. Returns a deterministic score in [0.0, 1.0]

Agent callable signature:
    agent(observation: Observation) -> Action
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from environment import (
    Action,
    MedicalTriageEnv,
    Observation,
    Reward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskMeta:
    task_id: str
    name: str
    difficulty: str
    target_score: float
    description: str
    case_ids: list[str]


TASK_REGISTRY: dict[str, TaskMeta] = {
    "task_1": TaskMeta(
        task_id="task_1",
        name="Stable Patient Triage",
        difficulty="easy",
        target_score=0.60,
        description=(
            "Triage 5 clearly stable patients with minor injuries or benign presentations. "
            "All cases have normal or near-normal vitals, no comorbidities, and obvious clinical "
            "diagnoses. The agent must correctly assign ESI 4-5 and recommend appropriate "
            "outpatient-style actions."
        ),
        case_ids=["PT-001", "PT-002", "PT-003", "PT-004", "PT-005"],
    ),
    "task_2": TaskMeta(
        task_id="task_2",
        name="Mixed Urgency Triage",
        difficulty="medium",
        target_score=0.40,
        description=(
            "Triage 5 mixed-acuity cases including 2 with ambiguous initial presentations. "
            "Cases include chest pain, possible sepsis, renal colic, appendicitis, and a possible "
            "subarachnoid haemorrhage. The agent must distinguish true emergencies (ESI 1-2) from "
            "urgent cases (ESI 3) and recommend time-sensitive interventions."
        ),
        case_ids=["PT-006", "PT-008", "PT-009", "PT-010", "PT-011"],
    ),
    "task_3": TaskMeta(
        task_id="task_3",
        name="Complex Multi-System Critical Triage",
        difficulty="hard",
        target_score=0.20,
        description=(
            "Triage 5 complex critical care cases with misleading vitals, multi-system pathology, "
            "life-threatening conditions masked by deceptive clinical features, and challenging "
            "treatment decisions (e.g., drug toxicity, drug allergies, anticoagulation conflicts). "
            "All require ESI 1-2. Missing critical flags incurs significant penalties."
        ),
        case_ids=["PT-012", "PT-014", "PT-015", "PT-019", "PT-024"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner - shared logic
# ─────────────────────────────────────────────────────────────────────────────

AgentFn = Callable[[Observation], Action]


def run_episode(
    agent: AgentFn,
    case_ids: list[str],
    shuffle: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Run a single episode with the given agent and return structured results.

    Args:
        agent:    Callable that maps Observation -> Action.
        case_ids: Patient IDs to include in the episode (fixed order if shuffle=False).
        shuffle:  Whether to shuffle case order (False for deterministic grading).
        seed:     RNG seed for reproducibility.
        verbose:  Print per-step info.

    Returns:
        dict with keys: scores, mean_score, actions, rewards, info_list, wall_time_s
    """
    env = MedicalTriageEnv(seed=seed)
    obs = env.reset(case_ids=case_ids, shuffle=shuffle)

    scores: list[float] = []
    actions: list[dict] = []
    rewards: list[dict] = []
    infos: list[dict] = []

    t0 = time.monotonic()
    done = False

    while not done:
        action = agent(obs)
        next_obs, reward, done, info = env.step(action)

        scores.append(reward.score)
        actions.append(action.model_dump())
        rewards.append(reward.model_dump())
        infos.append(info)

        if verbose:
            print(
                f"  [{info['case_id']}] "
                f"assigned={action.triage_level} correct={info['correct_triage_level']} "
                f"score={reward.score:.3f}"
            )

        obs = next_obs  # None when done

    wall_time = time.monotonic() - t0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "actions": actions,
        "rewards": rewards,
        "info_list": infos,
        "wall_time_s": round(wall_time, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual graders - each returns float in [0.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_1(agent: AgentFn, verbose: bool = False) -> float:
    """
    TASK 1 - Easy: Triage 5 clearly stable patients.
    Target score: >= 0.85

    Grading:
    - Mean episode reward is the primary metric.
    - Normalized to [0, 1] using a sigmoid-like clamp anchored at target.
    - Deterministic: fixed case order, fixed seed.
    """
    meta = TASK_REGISTRY["task_1"]
    result = run_episode(
        agent=agent,
        case_ids=meta.case_ids,
        shuffle=False,
        seed=42,
        verbose=verbose,
    )
    raw = result["mean_score"]
    # Normalize: score below 0 is 0, at or above 1.0 is 1.0
    graded = max(0.0, min(1.0, raw))
    if verbose:
        print(f"Task 1 raw mean={raw:.4f}  graded={graded:.4f}  target={meta.target_score}")
    return round(graded, 4)


def grade_task_2(agent: AgentFn, verbose: bool = False) -> float:
    """
    TASK 2 - Medium: 5 mixed cases including 2 ambiguous ones.
    Target score: >= 0.70

    Grading:
    - Mean reward weighted by difficulty:
        ambiguous cases (PT-009, PT-011) weighted 1.5x
        clear cases weighted 1.0x
    - Final score is the weighted mean, clamped to [0, 1].
    """
    meta = TASK_REGISTRY["task_2"]
    result = run_episode(
        agent=agent,
        case_ids=meta.case_ids,
        shuffle=False,
        seed=42,
        verbose=verbose,
    )

    # Map case ID -> score
    case_scores: dict[str, float] = {}
    for info, score in zip(result["info_list"], result["scores"]):
        case_scores[info["case_id"]] = score

    # Ambiguous cases worth 1.5x
    ambiguous = {"PT-009", "PT-011"}
    weights = {cid: (1.5 if cid in ambiguous else 1.0) for cid in meta.case_ids}

    numerator = sum(case_scores.get(cid, 0.0) * w for cid, w in weights.items())
    denominator = sum(weights.values())
    graded = max(0.0, min(1.0, numerator / denominator))

    if verbose:
        print(
            f"Task 2 weighted score={graded:.4f}  "
            f"unweighted mean={result['mean_score']:.4f}  "
            f"target={meta.target_score}"
        )
    return round(graded, 4)


def grade_task_3(agent: AgentFn, verbose: bool = False) -> float:
    """
    TASK 3 - Hard: 5 complex multi-system critical cases.
    Target score: >= 0.55

    Grading:
    - Base: mean reward.
    - Critical flag bonus: +0.05 per case where ALL critical flags are caught.
    - Critical flag miss penalty: -0.05 per case where critical flags are entirely missed.
    - Final score clamped to [0, 1].
    """
    meta = TASK_REGISTRY["task_3"]
    result = run_episode(
        agent=agent,
        case_ids=meta.case_ids,
        shuffle=False,
        seed=42,
        verbose=verbose,
    )

    base = result["mean_score"]
    adjustments = 0.0

    for info, action_dict in zip(result["info_list"], result["actions"]):
        gt_flags = info["correct_critical_flags"]
        pred_flags = action_dict.get("critical_flags", [])
        if gt_flags:
            # Measure token-level recall
            gt_tokens = set(" ".join(f.lower() for f in gt_flags).split())
            pred_tokens = set(" ".join(f.lower() for f in pred_flags).split())
            recall = len(gt_tokens & pred_tokens) / max(len(gt_tokens), 1)
            if recall == 1.0:
                adjustments += 0.05  # full flag bonus
            elif recall == 0.0:
                adjustments -= 0.05  # entire miss penalty

    graded = max(0.0, min(1.0, base + adjustments))

    if verbose:
        print(
            f"Task 3 base={base:.4f}  adjustments={adjustments:+.4f}  "
            f"graded={graded:.4f}  target={meta.target_score}"
        )
    return round(graded, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all tasks
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tasks(agent: AgentFn, verbose: bool = True) -> dict[str, float]:
    """
    Execute all three tasks and return a mapping of task_id -> grade.

    Args:
        agent:   Agent callable (Observation -> Action).
        verbose: Print per-task progress.

    Returns:
        {"task_1": float, "task_2": float, "task_3": float}
    """
    graders = {
        "task_1": grade_task_1,
        "task_2": grade_task_2,
        "task_3": grade_task_3,
    }
    results: dict[str, float] = {}
    for task_id, grader in graders.items():
        meta = TASK_REGISTRY[task_id]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {meta.name}  [{meta.difficulty.upper()}]")
            print(f"  Target: >= {meta.target_score}")
            print(f"  Cases:  {meta.case_ids}")
            print("=" * 60)
        score = grader(agent, verbose=verbose)
        results[task_id] = score
        passed = "PASS" if score >= meta.target_score else "FAIL"
        if verbose:
            print(f"  -> Final grade: {score:.4f}  [{passed}]")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo - smoke test with a deterministic baseline agent
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from environment import Action

    def _baseline_agent(obs: Observation) -> Action:
        """Naive heuristic: triage based on oxygen saturation and HR."""
        hr = obs.vitals.heart_rate
        spo2 = obs.vitals.oxygen_saturation
        gcs = obs.vitals.glasgow_coma_scale
        sbp = obs.vitals.systolic_bp

        if gcs < 13 or spo2 < 90 or sbp < 90:
            level = 1
            flags = ["critical deterioration detected"]
            actions = ["IV access", "oxygen", "emergency consult", "continuous monitoring"]
        elif spo2 < 94 or hr > 120 or sbp < 100:
            level = 2
            flags = ["hemodynamic compromise possible"]
            actions = ["IV access", "labs", "cardiac monitoring", "specialist consult"]
        elif hr > 100 or obs.vitals.temperature_celsius > 38.5:
            level = 3
            flags = []
            actions = ["labs", "IV access", "clinical assessment"]
        elif obs.vitals.pain_score > 5:
            level = 4
            flags = []
            actions = ["analgesia", "assessment", "targeted workup"]
        else:
            level = 5
            flags = []
            actions = ["symptomatic treatment", "discharge with follow-up"]

        return Action(
            triage_level=level,
            recommended_actions=actions,
            critical_flags=flags,
            reasoning="Baseline heuristic agent based on vitals thresholds.",
        )

    scores = run_all_tasks(_baseline_agent, verbose=True)
    print("\n" + "=" * 60)
    print("BASELINE AGENT - FINAL SCORES")
    print("=" * 60)
    for tid, score in scores.items():
        meta = TASK_REGISTRY[tid]
        status = "PASS" if score >= meta.target_score else "FAIL"
        print(f"  {tid}: {score:.4f}  [{status}]  (target >= {meta.target_score})")
