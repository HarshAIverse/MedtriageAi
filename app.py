"""
app.py — MedTriage AI FastAPI Server
Serves the static dashboard and exposes the OpenEnv REST API.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import Action, MedicalTriageEnv, PATIENT_CASES
from tasks import TASK_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedTriage AI — OpenEnv",
    description="Medical Triage & Clinical Decision Support RL environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = MedicalTriageEnv(seed=42)

# Mount static files (dashboard)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    case_ids: Optional[List[str]] = None
    shuffle: bool = False
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: Dict[str, Any]


class RunTaskRequest(BaseModel):
    task_id: str = "task_1"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive dashboard if available, otherwise return API info."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    # Fallback: clean JSON API info (e.g. when dashboard not bundled)
    return {
        "status": "MedTriage AI running",
        "version": "1.0.0",
        "description": "Medical Triage & Clinical Decision Support — OpenEnv",
        "endpoints": {
            "health":    "GET  /health",
            "reset":     "POST /reset",
            "step":      "POST /step",
            "state":     "GET  /state",
            "tasks":     "GET  /tasks",
            "cases":     "GET  /cases",
            "run_task":  "POST /run_task",
            "docs":      "GET  /docs",
        },
        "inference": "python inference.py",
        "validate":  "python validate.py",
    }


@app.get("/health")
async def health():
    """Health-check endpoint polled by the dashboard every 5 s."""
    return {"status": "ok", "version": "1.0.0", "timestamp": time.time()}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Optionally specify case_ids to run only specific patients.
    """
    try:
        obs = env.reset(case_ids=req.case_ids, shuffle=req.shuffle)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    """
    Submit a triage action for the current patient.
    Returns: observation (next), reward, done flag, info dict.
    """
    try:
        action_data = req.action

        # Robust coercion (same as inference.py)
        raw_level = action_data.get("triage_level", 3)
        try:
            level = max(1, min(5, int(float(str(raw_level).strip()))))
        except (ValueError, TypeError):
            level = 3

        raw_actions = action_data.get("recommended_actions") or []
        if isinstance(raw_actions, str):
            raw_actions = [a.strip() for a in raw_actions.split(",") if a.strip()]
        actions = [str(a) for a in raw_actions if a] or ["clinical assessment"]

        raw_flags = action_data.get("critical_flags") or []
        if isinstance(raw_flags, str):
            raw_flags = [f.strip() for f in raw_flags.split(",") if f.strip()]
        flags = [str(f) for f in raw_flags if f]

        action = Action(
            triage_level=level,
            recommended_actions=actions,
            critical_flags=flags,
            reasoning=str(action_data.get("reasoning") or ""),
        )

        next_obs, reward, done, info = env.step(action)

        return {
            "observation": next_obs.model_dump() if next_obs else None,
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
async def state():
    """Return the current full environment state."""
    return env.state()


@app.get("/cases")
async def list_cases():
    """List all 25 patient cases, grouped by difficulty."""
    by_diff: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
    for c in PATIENT_CASES:
        by_diff.setdefault(c.difficulty, []).append(c.patient_id)
    return {
        "total": len(PATIENT_CASES),
        "cases_by_difficulty": by_diff,
        "all_ids": [c.patient_id for c in PATIENT_CASES],
    }


@app.get("/case/{patient_id}")
async def get_case(patient_id: str):
    """Return metadata for a specific patient case."""
    case = env.get_case_by_id(patient_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {patient_id!r} not found.")
    return {
        "patient_id": case.patient_id,
        "vitals": case.vitals,
        "symptoms": case.symptoms,
        "history": case.history,
        "additional_context": case.additional_context,
        "difficulty": case.difficulty,
        "correct_triage_level": case.correct_triage_level,
        "correct_actions": case.correct_actions,
        "critical_flags": case.critical_flags,
        "explanation": case.explanation,
    }


@app.get("/tasks")
async def get_tasks():
    """Return the 3 task definitions with metadata."""
    tasks = []
    for tid, meta in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "name": meta.name,
            "difficulty": meta.difficulty,
            "target_score": meta.target_score,
            "description": meta.description,
            "case_ids": meta.case_ids,
        })
    return {"tasks": tasks}


@app.post("/run_task")
async def run_task(req: RunTaskRequest):
    """
    Run a full task episode using the built-in baseline heuristic agent.
    Returns per-step results and final grade.
    """
    task_id = req.task_id
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id!r}. Valid: {list(TASK_REGISTRY)}")

    meta = TASK_REGISTRY[task_id]

    # Simple heuristic baseline agent
    def heuristic_agent(obs):
        v = obs.vitals
        if v.glasgow_coma_scale < 13 or v.oxygen_saturation < 90 or v.systolic_bp < 90:
            return Action(triage_level=1,
                          recommended_actions=["IV access x2", "oxygen 100%", "emergency consult"],
                          critical_flags=["critical deterioration"],
                          reasoning="Heuristic: critical vitals")
        elif v.oxygen_saturation < 94 or v.heart_rate > 120 or v.systolic_bp < 100:
            return Action(triage_level=2,
                          recommended_actions=["IV access", "labs", "cardiac monitoring"],
                          critical_flags=[],
                          reasoning="Heuristic: compromised vitals")
        elif v.heart_rate > 100 or v.temperature_celsius > 38.5:
            return Action(triage_level=3,
                          recommended_actions=["labs", "IV access", "assessment"],
                          critical_flags=[],
                          reasoning="Heuristic: mild abnormality")
        elif v.pain_score > 5:
            return Action(triage_level=4,
                          recommended_actions=["analgesia", "assessment"],
                          critical_flags=[],
                          reasoning="Heuristic: pain")
        else:
            return Action(triage_level=5,
                          recommended_actions=["physical exam", "discharge instructions"],
                          critical_flags=[],
                          reasoning="Heuristic: stable")

    # Run episode
    t0 = time.monotonic()
    obs = env.reset(case_ids=meta.case_ids, shuffle=False)
    steps = []
    done = False

    while not done:
        action = heuristic_agent(obs)
        next_obs, reward, done, info = env.step(action)
        steps.append({
            "case_id": info["case_id"],
            "assigned_triage_level": action.triage_level,
            "correct_triage_level": info["correct_triage_level"],
            "score": reward.score,
            "reward_breakdown": reward.breakdown,
            "critical_flags_predicted": action.critical_flags,
            "critical_flags_correct": info["correct_critical_flags"],
            "difficulty": info["difficulty"],
        })
        obs = next_obs

    wall_time = time.monotonic() - t0
    scores = [s["score"] for s in steps]
    final_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "task_id": task_id,
        "task_name": meta.name,
        "difficulty": meta.difficulty,
        "target_score": meta.target_score,
        "final_score": final_score,
        "passed": final_score >= meta.target_score,
        "wall_time_s": round(wall_time, 2),
        "steps": steps,
        "agent": "heuristic_baseline",
    }
