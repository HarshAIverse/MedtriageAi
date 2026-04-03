"""
app.py - FastAPI wrapper for Medical Triage Environment
OpenEnv Submission - HarshAIverse/Meta

Endpoints:
    POST /reset   - Start a new episode
    POST /step    - Take an action in the current episode
    GET  /state   - Return full environment state
    GET  /        - Health check
    GET  /docs    - Auto-generated Swagger UI (FastAPI built-in)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import (
    Action,
    MedicalTriageEnv,
    Observation,
    Reward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan - create shared environment instance
# ─────────────────────────────────────────────────────────────────────────────

_env: Optional[MedicalTriageEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = MedicalTriageEnv(seed=42)
    yield
    _env = None


# ─────────────────────────────────────────────────────────────────────────────
# App instantiation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Triage Environment API",
    description=(
        "OpenEnv-compatible REST API for the Medical Triage & Clinical Decision Support "
        "environment. An AI agent acts as a triage nurse, processing patient cases and "
        "receiving reward signals for correct clinical decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    case_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of patient IDs to include in this episode. "
            "If omitted, all 25 cases are used."
        ),
        examples=[["PT-001", "PT-002", "PT-003", "PT-004", "PT-005"]],
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to randomise case order within the episode.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="RNG seed for reproducible shuffling. Ignored when shuffle=False.",
    )


class ResetResponse(BaseModel):
    observation: Observation
    message: str = "Episode started successfully."


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Optional[Observation] = Field(
        default=None,
        description="Next patient observation. None when episode is done.",
    )
    reward: Reward
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    state: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    environment: str = "medical-triage-env"
    total_cases_available: int = 25


class CasesResponse(BaseModel):
    cases_by_difficulty: dict[str, list[str]]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_env() -> MedicalTriageEnv:
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    return _env


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns environment status and version information.
    """
    return HealthResponse()


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    """
    Start a new episode.

    Optionally select specific patient cases by ID and control shuffling.
    Returns the first patient observation.
    """
    env = _get_env()

    # Re-seed if requested
    if request.seed is not None:
        import random
        env._rng = random.Random(request.seed)

    try:
        obs = env.reset(case_ids=request.case_ids, shuffle=request.shuffle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: StepRequest) -> StepResponse:
    """
    Submit a triage action for the current patient.

    The action must include:
    - `triage_level` (1-5 ESI priority)
    - `recommended_actions` (list of clinical interventions)
    - `critical_flags` (life-threatening conditions, if any)

    Returns the next observation, reward, done flag, and diagnostic info.
    """
    env = _get_env()

    if not env._episode_cases:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    if env._current_index >= len(env._episode_cases):
        raise HTTPException(
            status_code=400,
            detail="Episode is complete. Call POST /reset to start a new episode.",
        )

    next_obs, reward, done, info = env.step(request.action)

    return StepResponse(
        observation=next_obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse, tags=["Environment"])
async def get_state() -> StateResponse:
    """
    Retrieve the full current environment state.

    Returns episode progress, scores to date, current patient ID,
    last action taken, and last reward received.
    """
    env = _get_env()
    return StateResponse(state=env.state())


@app.get("/cases", response_model=CasesResponse, tags=["Environment"])
async def list_cases() -> CasesResponse:
    """
    List all available patient case IDs grouped by difficulty.

    Useful for constructing custom episodes or inspecting the case library.
    """
    env = _get_env()
    by_diff = env.case_ids_by_difficulty
    total = sum(len(v) for v in by_diff.values())
    return CasesResponse(cases_by_difficulty=by_diff, total=total)


@app.get("/case/{patient_id}", tags=["Environment"])
async def get_case_info(patient_id: str) -> dict[str, Any]:
    """
    Retrieve metadata for a specific patient case (without revealing ground truth labels).

    Returns patient ID, difficulty, and chief complaint only.
    """
    env = _get_env()
    case = env.get_case_by_id(patient_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Patient ID '{patient_id}' not found.")
    return {
        "patient_id": case.patient_id,
        "difficulty": case.difficulty,
        "chief_complaint": case.history.get("chief_complaint", ""),
        "age": case.history.get("age"),
        "sex": case.history.get("sex"),
    }
