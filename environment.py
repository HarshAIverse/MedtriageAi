"""
Medical Triage & Clinical Decision Support Environment
OpenEnv Submission - HarshAIverse/Meta

An RL environment where an AI agent acts as a triage nurse, processing
incoming patient cases and making evidence-based clinical decisions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class Vitals(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in beats per minute")
    systolic_bp: int = Field(..., description="Systolic blood pressure in mmHg")
    diastolic_bp: int = Field(..., description="Diastolic blood pressure in mmHg")
    respiratory_rate: int = Field(..., description="Respiratory rate in breaths per minute")
    temperature_celsius: float = Field(..., description="Body temperature in Celsius")
    oxygen_saturation: float = Field(..., description="SpO2 percentage (0-100)")
    glasgow_coma_scale: int = Field(..., ge=3, le=15, description="GCS score (3-15)")
    pain_score: int = Field(..., ge=0, le=10, description="Pain scale (0-10)")


class PatientHistory(BaseModel):
    age: int = Field(..., description="Patient age in years")
    sex: str = Field(..., description="Biological sex: M or F")
    chief_complaint: str = Field(..., description="Primary reason for presenting")
    onset_hours: float = Field(..., description="Hours since symptom onset")
    past_medical_history: list[str] = Field(default_factory=list, description="Relevant PMH diagnoses")
    current_medications: list[str] = Field(default_factory=list, description="Active medications")
    allergies: list[str] = Field(default_factory=list, description="Known allergies")
    surgical_history: list[str] = Field(default_factory=list, description="Past surgeries")


class Observation(BaseModel):
    """Observation returned to the agent at each timestep."""
    patient_id: str = Field(..., description="Unique patient case identifier")
    vitals: Vitals = Field(..., description="Current vital signs")
    symptoms: list[str] = Field(..., description="List of presenting symptoms")
    history: PatientHistory = Field(..., description="Patient background and history")
    additional_context: str = Field(default="", description="Nurse notes / EMS report / additional findings")
    timestamp: str = Field(..., description="ISO-8601 timestamp of triage presentation")
    case_index: int = Field(..., description="Index of current case in episode (0-based)")
    total_cases: int = Field(..., description="Total cases in this episode")


class Action(BaseModel):
    """Action taken by the agent for a patient case."""
    triage_level: int = Field(
        ..., ge=1, le=5,
        description=(
            "ESI triage priority: "
            "1=Immediate (life-threatening), "
            "2=Emergent (high risk), "
            "3=Urgent (serious), "
            "4=Less urgent (stable), "
            "5=Non-urgent (minor)"
        )
    )
    recommended_actions: list[str] = Field(
        ..., min_length=1,
        description="Ordered list of recommended clinical actions (e.g. 'IV access', 'ECG', 'oxygen therapy')"
    )
    critical_flags: list[str] = Field(
        default_factory=list,
        description="Life-threatening conditions identified (empty if none suspected)"
    )
    reasoning: str = Field(
        default="",
        description="Optional clinical reasoning / differential diagnosis notes"
    )


class Reward(BaseModel):
    """Reward signal returned after each action."""
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized reward in [0, 1]")
    breakdown: dict[str, float] = Field(..., description="Score component breakdown")
    feedback: str = Field(default="", description="Clinical feedback on the action")


# ─────────────────────────────────────────────────────────────────────────────
# Patient Case Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientCase:
    patient_id: str
    vitals: dict
    symptoms: list[str]
    history: dict
    additional_context: str
    correct_triage_level: int
    correct_actions: list[str]           # canonical actions (ground truth)
    critical_flags: list[str]            # life-threatening flags if any
    difficulty: str                      # "easy" | "medium" | "hard"
    explanation: str                     # clinical rationale (used in feedback)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Patient Case Library  (25 cases)
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_CASES: list[PatientCase] = [

    # ── EASY CASES (clearly stable, ESI 4-5) ────────────────────────────────

    PatientCase(
        patient_id="PT-001",
        vitals=dict(heart_rate=72, systolic_bp=118, diastolic_bp=76,
                    respiratory_rate=14, temperature_celsius=36.8,
                    oxygen_saturation=99.0, glasgow_coma_scale=15, pain_score=2),
        symptoms=["small laceration on right forearm", "mild bleeding controlled by pressure"],
        history=dict(age=24, sex="M", chief_complaint="cut arm",
                     onset_hours=0.5, past_medical_history=[],
                     current_medications=[], allergies=[], surgical_history=[]),
        additional_context="Tetanus immunization up to date. Wound ~2 cm, no tendon involvement.",
        correct_triage_level=4,
        correct_actions=["wound cleaning and irrigation", "suture or steri-strip closure",
                         "tetanus prophylaxis assessment", "discharge with wound care instructions"],
        critical_flags=[],
        difficulty="easy",
        explanation="Minor laceration, all vitals normal, no systemic involvement. ESI 4 appropriate."
    ),

    PatientCase(
        patient_id="PT-002",
        vitals=dict(heart_rate=80, systolic_bp=122, diastolic_bp=80,
                    respiratory_rate=16, temperature_celsius=36.6,
                    oxygen_saturation=98.0, glasgow_coma_scale=15, pain_score=3),
        symptoms=["right ankle pain", "mild swelling", "difficulty bearing weight"],
        history=dict(age=19, sex="F", chief_complaint="twisted ankle playing basketball",
                     onset_hours=1.0, past_medical_history=[],
                     current_medications=[], allergies=[], surgical_history=[]),
        additional_context="Ottawa ankle rules: tenderness at posterior tip of lateral malleolus.",
        correct_triage_level=4,
        correct_actions=["X-ray ankle (Ottawa rules positive)", "ice and elevation",
                         "analgesia (NSAIDs)", "orthopedic follow-up if fracture confirmed"],
        critical_flags=[],
        difficulty="easy",
        explanation="Ankle sprain/possible fracture. Stable vitals. ESI 4 - needs 1 resource (X-ray)."
    ),

    PatientCase(
        patient_id="PT-003",
        vitals=dict(heart_rate=68, systolic_bp=114, diastolic_bp=72,
                    respiratory_rate=13, temperature_celsius=36.5,
                    oxygen_saturation=99.0, glasgow_coma_scale=15, pain_score=1),
        symptoms=["sore throat", "mild fever", "difficulty swallowing solid food"],
        history=dict(age=16, sex="F", chief_complaint="sore throat for 2 days",
                     onset_hours=48.0, past_medical_history=[],
                     current_medications=[], allergies=["penicillin"],
                     surgical_history=[]),
        additional_context="Centor score 3: exudate, tender anterior cervical nodes, no cough, fever 37.8°C.",
        correct_triage_level=5,
        correct_actions=["throat swab for rapid strep test", "symptomatic analgesia",
                         "prescribe appropriate antibiotic (avoid penicillin)",
                         "discharge with return precautions"],
        critical_flags=[],
        difficulty="easy",
        explanation="Likely strep pharyngitis. Stable, non-severe. ESI 5 - minor problem, 0 resources."
    ),

    PatientCase(
        patient_id="PT-004",
        vitals=dict(heart_rate=76, systolic_bp=120, diastolic_bp=78,
                    respiratory_rate=15, temperature_celsius=37.0,
                    oxygen_saturation=98.5, glasgow_coma_scale=15, pain_score=2),
        symptoms=["lower back pain", "no radiation to legs", "pain after lifting heavy boxes"],
        history=dict(age=35, sex="M", chief_complaint="back pain",
                     onset_hours=6.0, past_medical_history=[],
                     current_medications=["ibuprofen PRN"], allergies=[],
                     surgical_history=[]),
        additional_context="No saddle anesthesia, no bowel/bladder dysfunction, negative straight-leg raise.",
        correct_triage_level=5,
        correct_actions=["oral analgesia (NSAIDs + muscle relaxant)", "physical therapy referral",
                         "return precautions for cauda equina symptoms", "discharge"],
        critical_flags=[],
        difficulty="easy",
        explanation="Mechanical low-back pain. No red flags for cauda equina or fracture. ESI 5."
    ),

    PatientCase(
        patient_id="PT-005",
        vitals=dict(heart_rate=74, systolic_bp=116, diastolic_bp=74,
                    respiratory_rate=14, temperature_celsius=36.9,
                    oxygen_saturation=99.0, glasgow_coma_scale=15, pain_score=1),
        symptoms=["runny nose", "mild cough", "congestion", "sneezing"],
        history=dict(age=28, sex="F", chief_complaint="cold symptoms for 3 days",
                     onset_hours=72.0, past_medical_history=[],
                     current_medications=[], allergies=[],
                     surgical_history=[]),
        additional_context="No shortness of breath, no chest pain, recovering well. Seeking sick note.",
        correct_triage_level=5,
        correct_actions=["symptomatic treatment (decongestants, fluids)", "sick note",
                         "return if fever > 38.5°C or symptoms worsen",
                         "COVID-19 self-testing recommended"],
        critical_flags=[],
        difficulty="easy",
        explanation="Viral upper respiratory tract infection. All vitals normal. ESI 5."
    ),

    # ── MEDIUM CASES (mixed / ambiguous, ESI 2-4) ───────────────────────────

    PatientCase(
        patient_id="PT-006",
        vitals=dict(heart_rate=102, systolic_bp=148, diastolic_bp=92,
                    respiratory_rate=18, temperature_celsius=37.2,
                    oxygen_saturation=96.0, glasgow_coma_scale=15, pain_score=6),
        symptoms=["crushing chest pressure", "radiation to left arm", "diaphoresis",
                  "onset at rest 30 minutes ago"],
        history=dict(age=58, sex="M", chief_complaint="chest pain",
                     onset_hours=0.5, past_medical_history=["hypertension", "hyperlipidemia"],
                     current_medications=["amlodipine", "atorvastatin"],
                     allergies=[], surgical_history=[]),
        additional_context="TIMI risk score: 4. No prior cardiac history. Pain not positional.",
        correct_triage_level=2,
        correct_actions=["12-lead ECG immediately", "IV access x2", "aspirin 324mg PO",
                         "serial troponins", "cardiology consult", "continuous cardiac monitoring",
                         "oxygen if SpO2 < 94%"],
        critical_flags=["possible STEMI/NSTEMI", "acute coronary syndrome"],
        difficulty="medium",
        explanation="Classic ACS presentation. TIMI 4 intermediate-high risk. ESI 2 - emergent."
    ),

    PatientCase(
        patient_id="PT-007",
        vitals=dict(heart_rate=88, systolic_bp=126, diastolic_bp=82,
                    respiratory_rate=22, temperature_celsius=38.4,
                    oxygen_saturation=94.0, glasgow_coma_scale=15, pain_score=5),
        symptoms=["productive cough", "pleuritic chest pain", "fever", "malaise"],
        history=dict(age=67, sex="F", chief_complaint="cough and fever for 4 days",
                     onset_hours=96.0, past_medical_history=["type 2 diabetes", "COPD"],
                     current_medications=["metformin", "tiotropium", "salbutamol PRN"],
                     allergies=["sulfa drugs"], surgical_history=[]),
        additional_context="Decreased breath sounds right lower lobe. CRB-65 score: 2.",
        correct_triage_level=3,
        correct_actions=["chest X-ray", "sputum culture", "blood cultures x2",
                         "CBC, BMP, CRP", "IV antibiotics (community-acquired pneumonia protocol)",
                         "oxygen therapy to maintain SpO2 > 94%", "strict I&O"],
        critical_flags=["pneumonia with borderline sepsis risk"],
        difficulty="medium",
        explanation="CAP in elderly diabetic COPD patient, SpO2 94%, CRB-65 2 - moderate severity. ESI 3."
    ),

    PatientCase(
        patient_id="PT-008",
        vitals=dict(heart_rate=94, systolic_bp=136, diastolic_bp=88,
                    respiratory_rate=19, temperature_celsius=37.8,
                    oxygen_saturation=97.0, glasgow_coma_scale=15, pain_score=7),
        symptoms=["severe right-sided flank pain", "nausea", "vomiting", "hematuria"],
        history=dict(age=42, sex="M", chief_complaint="sudden severe flank pain",
                     onset_hours=2.0, past_medical_history=["nephrolithiasis (previous episode)"],
                     current_medications=[], allergies=["morphine"],
                     surgical_history=[]),
        additional_context="Pain is colicky, radiating to groin. No fever. CVA tenderness positive.",
        correct_triage_level=3,
        correct_actions=["IV access", "urinalysis and urine microscopy", "BMP (creatinine, electrolytes)",
                         "CT KUB non-contrast", "IV analgesia (ketorolac - check allergy)",
                         "anti-emetics", "urology consult if stone > 6mm"],
        critical_flags=[],
        difficulty="medium",
        explanation="Classic renal colic, prior history. Note morphine allergy. ESI 3."
    ),

    PatientCase(
        patient_id="PT-009",
        vitals=dict(heart_rate=110, systolic_bp=104, diastolic_bp=68,
                    respiratory_rate=20, temperature_celsius=38.9,
                    oxygen_saturation=95.0, glasgow_coma_scale=14, pain_score=4),
        symptoms=["confusion", "dysuria", "frequency", "suprapubic tenderness", "rigors"],
        history=dict(age=78, sex="F", chief_complaint="confusion and fever",
                     onset_hours=8.0, past_medical_history=["type 2 diabetes", "CKD stage 3",
                                                              "recurrent UTIs"],
                     current_medications=["insulin glargine", "lisinopril", "furosemide"],
                     allergies=[], surgical_history=["hysterectomy"]),
        additional_context="GCS dropped from baseline 15 to 14. Urine: cloudy, foul-smelling, dipstick: LE+++ nitrites+.",
        correct_triage_level=2,
        correct_actions=["blood cultures x2 before antibiotics", "IV access x2",
                         "CBC, BMP, lactate, urinalysis, urine culture",
                         "IV antibiotics (broad-spectrum)", "IV fluid resuscitation",
                         "continuous monitoring", "sepsis protocol activation"],
        critical_flags=["urosepsis", "sepsis in elderly diabetic", "altered mental status"],
        difficulty="medium",
        explanation="Urosepsis meets SIRS criteria in elderly diabetic. Altered GCS. ESI 2 - emergent."
    ),

    PatientCase(
        patient_id="PT-010",
        vitals=dict(heart_rate=84, systolic_bp=128, diastolic_bp=84,
                    respiratory_rate=16, temperature_celsius=36.7,
                    oxygen_saturation=98.0, glasgow_coma_scale=15, pain_score=5),
        symptoms=["right lower quadrant pain", "anorexia", "nausea", "low-grade fever",
                  "pain started periumbilically and migrated to RLQ"],
        history=dict(age=22, sex="F", chief_complaint="abdominal pain",
                     onset_hours=16.0, past_medical_history=[],
                     current_medications=[], allergies=[],
                     surgical_history=[]),
        additional_context="McBurney's point tenderness, Rovsing's sign positive, Alvarado score 7. LMP 14 days ago.",
        correct_triage_level=3,
        correct_actions=["IV access", "CBC (WBC, differential)", "CMP, CRP",
                         "urinalysis (exclude pyelonephritis)", "serum beta-hCG",
                         "ultrasound abdomen/pelvis", "surgical consult",
                         "NPO status", "IV fluid and analgesia"],
        critical_flags=["suspected appendicitis - perforation risk"],
        difficulty="medium",
        explanation="Alvarado 7 = high appendicitis probability. Must rule out ectopic pregnancy. ESI 3."
    ),

    PatientCase(
        patient_id="PT-011",
        vitals=dict(heart_rate=78, systolic_bp=142, diastolic_bp=88,
                    respiratory_rate=15, temperature_celsius=37.1,
                    oxygen_saturation=98.0, glasgow_coma_scale=15, pain_score=3),
        symptoms=["sudden severe headache", "'worst headache of my life'",
                  "neck stiffness", "photophobia"],
        history=dict(age=38, sex="M", chief_complaint="thunderclap headache",
                     onset_hours=1.0, past_medical_history=["migraines (different character)"],
                     current_medications=["sumatriptan PRN"], allergies=[],
                     surgical_history=[]),
        additional_context="Patient reports this headache reached maximum intensity within seconds. Kernig's sign equivocal.",
        correct_triage_level=2,
        correct_actions=["urgent non-contrast CT head", "lumbar puncture if CT negative",
                         "IV access", "strict pain monitoring",
                         "neurology/neurosurgery consult", "NPO"],
        critical_flags=["subarachnoid hemorrhage must be excluded",
                        "thunderclap headache - neurovascular emergency"],
        difficulty="medium",
        explanation="Thunderclap headache with meningismus - SAH until proven otherwise. ESI 2."
    ),

    # ── HARD CASES (complex, multi-system, misleading, ESI 1-3) ─────────────

    PatientCase(
        patient_id="PT-012",
        vitals=dict(heart_rate=52, systolic_bp=88, diastolic_bp=56,
                    respiratory_rate=10, temperature_celsius=35.1,
                    oxygen_saturation=91.0, glasgow_coma_scale=10, pain_score=1),
        symptoms=["unresponsive to verbal stimuli", "found on bathroom floor",
                  "empty pill bottles nearby", "meiotic pupils"],
        history=dict(age=31, sex="M", chief_complaint="unresponsive",
                     onset_hours=0.25, past_medical_history=["opioid use disorder"],
                     current_medications=["buprenorphine"],
                     allergies=[], surgical_history=[]),
        additional_context="EMS report: naloxone 0.4mg IM given en route - partial response. GCS improved from 6 to 10.",
        correct_triage_level=1,
        correct_actions=["repeat naloxone IV/IM titrated to response", "bag-valve-mask ventilation",
                         "IV access x2", "oxygen 100% non-rebreather",
                         "continuous pulse oximetry and cardiac monitoring",
                         "ABG", "blood glucose", "toxicology screen",
                         "ICU admission", "psychiatric evaluation once stabilized"],
        critical_flags=["opioid overdose with respiratory depression",
                        "GCS 10 - airway at risk", "hypothermia"],
        difficulty="hard",
        explanation="Opioid overdose partial naloxone response. Airway compromise imminent. ESI 1."
    ),

    PatientCase(
        patient_id="PT-013",
        vitals=dict(heart_rate=118, systolic_bp=82, diastolic_bp=50,
                    respiratory_rate=28, temperature_celsius=39.8,
                    oxygen_saturation=88.0, glasgow_coma_scale=12, pain_score=6),
        symptoms=["fever", "difficulty breathing", "productive cough - rusty sputum",
                  "confusion", "mottled skin"],
        history=dict(age=71, sex="M", chief_complaint="severe breathing difficulty",
                     onset_hours=12.0, past_medical_history=["CKD stage 4", "heart failure (EF 35%)",
                                                               "atrial fibrillation"],
                     current_medications=["warfarin", "digoxin", "furosemide", "carvedilol"],
                     allergies=["contrast dye - renal protocol required"],
                     surgical_history=["CABG 10 years ago"]),
        additional_context=(
            "INR pending. Chest XR: bilateral infiltrates. NT-proBNP elevated. "
            "Creatinine 3.2 from baseline 2.1. Lactate 4.8 mmol/L."
        ),
        correct_triage_level=1,
        correct_actions=["immediate airway assessment - consider early intubation",
                         "high-flow oxygen", "IV access x2 - large bore",
                         "blood cultures x2", "IV antibiotics (septic shock protocol)",
                         "cautious fluid resuscitation (HF history - 250mL boluses)",
                         "vasopressors if BP unresponsive", "ICU admission",
                         "cardiology + pulmonology consult",
                         "renal-dose contrast protocol for imaging",
                         "check INR before invasive procedures"],
        critical_flags=["septic shock", "type 1 respiratory failure (SpO2 88%)",
                        "acute kidney injury on CKD",
                        "anticoagulation complexity (warfarin)",
                        "cardiogenic + septic shock overlap"],
        difficulty="hard",
        explanation=(
            "Septic shock with multi-organ dysfunction. Complex: AKI, HF, AF on warfarin, "
            "CKD, SpO2 88%. lactate 4.8. ESI 1 - immediate."
        )
    ),

    PatientCase(
        patient_id="PT-014",
        vitals=dict(heart_rate=96, systolic_bp=176, diastolic_bp=108,
                    respiratory_rate=22, temperature_celsius=37.3,
                    oxygen_saturation=97.0, glasgow_coma_scale=13, pain_score=4),
        symptoms=["sudden right arm weakness", "slurred speech", "facial droop on right",
                  "mild headache"],
        history=dict(age=64, sex="F", chief_complaint="weakness and speech difficulty",
                     onset_hours=1.25, past_medical_history=["hypertension", "type 2 diabetes",
                                                               "atrial fibrillation"],
                     current_medications=["apixaban", "metformin", "ramipril"],
                     allergies=[], surgical_history=[]),
        additional_context=(
            "NIHSS estimated 8. Last known well 75 minutes ago. "
            "Patient on DOAC - standard thrombolysis protocol differs."
        ),
        correct_triage_level=1,
        correct_actions=["activate stroke code / FAST protocol",
                         "non-contrast CT head stat",
                         "CT angiography head and neck",
                         "glucose check immediately",
                         "IV access x2", "hold apixaban - anticoagulation status critical",
                         "neurology consult stat",
                         "assess tPA eligibility (DOAC - check reversal agent availability)",
                         "ECG", "strict BP management (target < 185/110 if tPA considered)",
                         "NPO"],
        critical_flags=["acute ischemic stroke - potential tPA candidate",
                        "DOAC anticoagulation complicates thrombolysis",
                        "within treatment window"],
        difficulty="hard",
        explanation=(
            "Acute ischemic stroke, NIHSS 8, within tPA window but on apixaban - "
            "requires immediate specialized protocol. ESI 1."
        )
    ),

    PatientCase(
        patient_id="PT-015",
        vitals=dict(heart_rate=44, systolic_bp=78, diastolic_bp=52,
                    respiratory_rate=12, temperature_celsius=35.8,
                    oxygen_saturation=93.0, glasgow_coma_scale=14, pain_score=3),
        symptoms=["syncope", "profound fatigue", "exertional dyspnea", "presyncope on standing"],
        history=dict(age=55, sex="M", chief_complaint="fainting episode",
                     onset_hours=0.5, past_medical_history=["hypothyroidism",
                                                              "previously 'structurally normal heart'"],
                     current_medications=["levothyroxine 100mcg", "metoprolol succinate 200mg"],
                     allergies=[], surgical_history=[]),
        additional_context=(
            "Misleading: patient says 'this happens when I'm anxious'. "
            "12-lead ECG: complete heart block (3rd degree), junctional escape rate 44bpm. "
            "Patient appears anxious and attributes all symptoms to panic attacks."
        ),
        correct_triage_level=1,
        correct_actions=["continuous cardiac monitoring", "transcutaneous pacing on standby",
                         "IV access x2", "atropine 0.5mg IV if haemodynamically unstable",
                         "cardiology consult for transvenous pacing",
                         "hold metoprolol immediately",
                         "check TSH (hypothyroidism contribution)",
                         "electrolytes, CBC", "ICU/CCU admission"],
        critical_flags=["complete heart block (3rd degree) - pacemaker emergency",
                        "haemodynamic compromise", "metoprolol may be precipitating factor"],
        difficulty="hard",
        explanation=(
            "3rd-degree heart block masked as anxiety/syncope. High-dose metoprolol "
            "potentially causative. Immediate pacing required. ESI 1."
        )
    ),

    PatientCase(
        patient_id="PT-016",
        vitals=dict(heart_rate=128, systolic_bp=92, diastolic_bp=62,
                    respiratory_rate=26, temperature_celsius=38.1,
                    oxygen_saturation=96.0, glasgow_coma_scale=15, pain_score=8),
        symptoms=["severe acute abdominal pain", "vomiting blood (bright red)",
                  "dizziness", "black tarry stools reported"],
        history=dict(age=49, sex="M", chief_complaint="vomiting blood",
                     onset_hours=3.0, past_medical_history=["peptic ulcer disease", "cirrhosis (Child-Pugh B)"],
                     current_medications=["propranolol", "spironolactone", "lactulose"],
                     allergies=[], surgical_history=["variceal banding x2"]),
        additional_context=(
            "Hematemesis approximately 500mL. Visually jaundiced. Spider angiomata present. "
            "Abdomen distended with shifting dullness. Vitals deteriorating."
        ),
        correct_triage_level=1,
        correct_actions=["two large-bore IV access immediately",
                         "aggressive IV fluid resuscitation",
                         "blood type and crossmatch - 4 units pRBC",
                         "emergent endoscopy preparation",
                         "IV proton pump inhibitor infusion",
                         "IV terlipressin or octreotide (variceal bleed protocol)",
                         "fresh frozen plasma (coagulopathy in cirrhosis)",
                         "gastroenterology / GI surgery consult",
                         "insert Sengstaken-Blakemore tube if endoscopy unavailable",
                         "ICU admission", "strict airway monitoring"],
        critical_flags=["upper GI hemorrhage - likely variceal",
                        "haemodynamic shock",
                        "cirrhosis with coagulopathy",
                        "aspiration risk during hematemesis"],
        difficulty="hard",
        explanation=(
            "Acute variceal haemorrhage in cirrhotic patient. Haemodynamic compromise, "
            "coagulopathy risk, prior banding. ESI 1 - immediate resuscitation."
        )
    ),

    PatientCase(
        patient_id="PT-017",
        vitals=dict(heart_rate=102, systolic_bp=158, diastolic_bp=96,
                    respiratory_rate=20, temperature_celsius=37.4,
                    oxygen_saturation=97.0, glasgow_coma_scale=15, pain_score=5),
        symptoms=["tearing interscapular back pain", "unequal blood pressure in arms",
                  "sudden onset", "sweating"],
        history=dict(age=52, sex="M", chief_complaint="severe sudden back pain",
                     onset_hours=0.75, past_medical_history=["hypertension (poorly controlled)",
                                                               "Marfan syndrome"],
                     current_medications=["amlodipine", "valsartan"], allergies=[],
                     surgical_history=[]),
        additional_context=(
            "BP right arm 158/96; left arm 128/82 - 30mmHg differential. "
            "CXR: widened mediastinum. Pain initially mimicked musculoskeletal."
        ),
        correct_triage_level=1,
        correct_actions=["immediate CT aorta with contrast",
                         "IV access x2 large bore",
                         "strict BP control (target SBP 100-120) - IV labetalol or esmolol",
                         "cardiothoracic surgery consult stat",
                         "NPO immediately",
                         "blood type and crossmatch",
                         "pain control avoiding anticoagulation",
                         "ICU admission", "continuous monitoring"],
        critical_flags=["aortic dissection (type A vs B - requires CT classification)",
                        "Marfan syndrome - high dissection risk",
                        "widened mediastinum"],
        difficulty="hard",
        explanation=(
            "Aortic dissection: tearing pain, BP differential, widened mediastinum, Marfan. "
            "Mimics musculoskeletal. ESI 1 - life-threatening vascular emergency."
        )
    ),

    PatientCase(
        patient_id="PT-018",
        vitals=dict(heart_rate=118, systolic_bp=94, diastolic_bp=60,
                    respiratory_rate=24, temperature_celsius=36.2,
                    oxygen_saturation=95.0, glasgow_coma_scale=15, pain_score=7),
        symptoms=["sudden onset pleuritic chest pain", "dyspnea", "one-sided leg swelling",
                  "recent long-haul flight (14 hours)"],
        history=dict(age=44, sex="F", chief_complaint="chest pain and breathing difficulty",
                     onset_hours=2.0, past_medical_history=["combined oral contraceptive pill use",
                                                              "obesity (BMI 34)"],
                     current_medications=["levonorgestrel/ethinylestradiol"],
                     allergies=["aspirin - causes urticaria"],
                     surgical_history=[]),
        additional_context=(
            "PERC score: 3 (positive). Wells PE score: 7.5 (high probability). "
            "D-dimer not appropriate given high clinical probability. "
            "Calf tender and swollen. SpO2 drops on exertion."
        ),
        correct_triage_level=2,
        correct_actions=["CT pulmonary angiography (CTPA) stat",
                         "IV access", "oxygen therapy",
                         "anticoagulation: LMWH or DOAC (avoid heparin if allergy considered)",
                         "CBC, coagulation panel, BMP",
                         "ECG (S1Q3T3 pattern?)", "echo if haemodynamically unstable",
                         "thrombophilia workup", "stop OCP immediately",
                         "thromboembolism consult"],
        critical_flags=["high-probability pulmonary embolism",
                        "DVT with haemodynamic compromise",
                        "aspirin allergy limits antiplatelet options"],
        difficulty="hard",
        explanation=(
            "Wells 7.5 = high-probability PE. OCP + obesity + long flight. "
            "Aspirin allergy noted. ESI 2 - emergent."
        )
    ),

    PatientCase(
        patient_id="PT-019",
        vitals=dict(heart_rate=134, systolic_bp=88, diastolic_bp=54,
                    respiratory_rate=30, temperature_celsius=40.2,
                    oxygen_saturation=90.0, glasgow_coma_scale=11, pain_score=3),
        symptoms=["altered consciousness", "high fever", "neck stiffness",
                  "non-blanching petechial rash", "photophobia", "headache"],
        history=dict(age=19, sex="M", chief_complaint="severe headache and rash",
                     onset_hours=6.0, past_medical_history=[],
                     current_medications=[], allergies=["penicillin - anaphylaxis"],
                     surgical_history=[]),
        additional_context=(
            "Petechial rash spreading rapidly. University dormitory resident. "
            "Meningococcal vaccination status unknown. "
            "Penicillin allergy: documented anaphylaxis."
        ),
        correct_triage_level=1,
        correct_actions=["immediate IV antibiotics - ceftriaxone 2g IV (penicillin cross-reactivity < 2%)",
                         "if true anaphylaxis to beta-lactams: chloramphenicol IV",
                         "dexamethasone IV before or with first antibiotic dose",
                         "blood cultures - do NOT delay antibiotics for culture",
                         "CT head before LP if GCS < 13 or papilledema",
                         "IV access x2", "strict isolation precautions",
                         "public health notification for contacts",
                         "ICU admission", "continuous haemodynamic monitoring"],
        critical_flags=["bacterial meningitis / meningococcal septicemia",
                        "non-blanching petechial rash - septicemia",
                        "penicillin anaphylaxis - antibiotic choice critical",
                        "GCS 11 - airway at risk"],
        difficulty="hard",
        explanation=(
            "Meningococcal disease with septicemia signs (petechiae). "
            "Penicillin anaphylaxis complicates treatment. Do not delay antibiotics. ESI 1."
        )
    ),

    PatientCase(
        patient_id="PT-020",
        vitals=dict(heart_rate=86, systolic_bp=130, diastolic_bp=82,
                    respiratory_rate=17, temperature_celsius=37.0,
                    oxygen_saturation=97.0, glasgow_coma_scale=15, pain_score=4),
        symptoms=["chronic knee pain", "mild joint stiffness", "no acute injury",
                  "gradually worsening over 6 months"],
        history=dict(age=62, sex="F", chief_complaint="knee pain",
                     onset_hours=4380.0,  # ~6 months
                     past_medical_history=["osteoarthritis", "hypertension", "CKD stage 2"],
                     current_medications=["ramipril", "topical diclofenac"],
                     allergies=[], surgical_history=["right knee arthroscopy 10 years ago"]),
        additional_context=(
            "Presenting to ED for pain flare after running out of medication. "
            "No systemic inflammation signs. No fever, no joint warmth, no effusion. "
            "Stable vitals entire visit."
        ),
        correct_triage_level=4,
        correct_actions=["X-ray knee", "analgesics (avoid systemic NSAIDs - CKD)",
                         "orthopedic outpatient referral",
                         "prescription renewal", "physiotherapy referral",
                         "primary care follow-up"],
        critical_flags=[],
        difficulty="medium",
        explanation=(
            "Osteoarthritis flare. Important: avoid systemic NSAIDs due to CKD. ESI 4. "
            "Misleading as patient is in ED - not an emergency."
        )
    ),

    PatientCase(
        patient_id="PT-021",
        vitals=dict(heart_rate=108, systolic_bp=100, diastolic_bp=65,
                    respiratory_rate=22, temperature_celsius=36.9,
                    oxygen_saturation=98.0, glasgow_coma_scale=15, pain_score=6),
        symptoms=["heavy vaginal bleeding", "cramping abdominal pain",
                  "dizziness on standing", "passing clots"],
        history=dict(age=27, sex="F", chief_complaint="heavy vaginal bleeding",
                     onset_hours=4.0, past_medical_history=[],
                     current_medications=[],
                     allergies=[], surgical_history=[]),
        additional_context=(
            "LMP 6 weeks ago. Home pregnancy test positive yesterday. "
            "No IUD, no known ectopic risk factors."
        ),
        correct_triage_level=2,
        correct_actions=["IV access x2", "serum beta-hCG quantitative",
                         "transvaginal ultrasound stat",
                         "CBC, blood type and screen (Rh status critical)",
                         "IV fluid resuscitation",
                         "Rh immunoglobulin if Rh-negative",
                         "OB/GYN consult stat",
                         "serial BP and HR monitoring"],
        critical_flags=["ectopic pregnancy must be excluded",
                        "haemodynamic instability",
                        "possible spontaneous abortion"],
        difficulty="medium",
        explanation=(
            "Threatened/ectopic pregnancy - haemodynamically borderline. "
            "ESI 2 - must rule out ectopic immediately."
        )
    ),

    PatientCase(
        patient_id="PT-022",
        vitals=dict(heart_rate=76, systolic_bp=132, diastolic_bp=84,
                    respiratory_rate=16, temperature_celsius=36.8,
                    oxygen_saturation=98.5, glasgow_coma_scale=15, pain_score=5),
        symptoms=["sudden painless vision loss in left eye",
                  "describes 'curtain coming down' over vision",
                  "no associated headache", "no pain on eye movement"],
        history=dict(age=68, sex="M", chief_complaint="sudden vision loss",
                     onset_hours=2.0, past_medical_history=["hypertension", "hyperlipidemia",
                                                              "type 2 diabetes"],
                     current_medications=["metformin", "lisinopril", "simvastatin"],
                     allergies=[], surgical_history=["cataract surgery left eye 2 years ago"]),
        additional_context=(
            "Ophthalmoscopy attempted - cup-to-disc ratio normal. "
            "Patient denies recent trauma. Vitals appear deceptively stable."
        ),
        correct_triage_level=2,
        correct_actions=["urgent ophthalmology consult - within 1 hour",
                         "ocular massage (if CRAO suspected)",
                         "intraocular pressure measurement",
                         "fluorescein angiography",
                         "complete ocular examination",
                         "blood glucose, CBC, ESR/CRP (giant cell arteritis?)",
                         "carotid Doppler (embolic source)",
                         "ECG and echocardiogram"],
        critical_flags=["retinal artery occlusion - vision loss is time-critical",
                        "possible giant cell arteritis (GCA)",
                        "embolic event - cardiac or carotid workup needed"],
        difficulty="hard",
        explanation=(
            "Sudden painless monocular vision loss = retinal artery occlusion until proven otherwise. "
            "Deceptively stable vitals. Treatment window < 4-6 hours. ESI 2."
        )
    ),

    PatientCase(
        patient_id="PT-023",
        vitals=dict(heart_rate=92, systolic_bp=144, diastolic_bp=90,
                    respiratory_rate=18, temperature_celsius=37.6,
                    oxygen_saturation=96.0, glasgow_coma_scale=15, pain_score=6),
        symptoms=["right-sided abdominal pain", "fever", "jaundice",
                  "nausea", "dark urine", "clay-colored stools"],
        history=dict(age=49, sex="F", chief_complaint="right-sided pain and jaundice",
                     onset_hours=24.0, past_medical_history=["gallstones (known)"],
                     current_medications=[], allergies=[],
                     surgical_history=[]),
        additional_context=(
            "Charcot's triad: fever + RUQ pain + jaundice. "
            "LFTs: elevated bilirubin 5.6, ALT 320, ALP 450. "
            "WBC 14.5. Ultrasound: dilated CBD 12mm, gallstones."
        ),
        correct_triage_level=2,
        correct_actions=["IV access", "IV fluids", "blood cultures x2",
                         "IV antibiotics (biliary spectrum)",
                         "MRCP or EUS for CBD stone assessment",
                         "GI/surgery consult for ERCP",
                         "CBC, LFTs, coagulation panel, lipase",
                         "NPO", "analgesia", "monitor for Reynold's pentad"],
        critical_flags=["ascending cholangitis",
                        "risk of progression to septic shock",
                        "Charcot's triad positive"],
        difficulty="hard",
        explanation=(
            "Ascending cholangitis (Charcot's triad + dilated CBD). "
            "Can progress rapidly to septic shock (Reynold's pentad). ESI 2."
        )
    ),

    PatientCase(
        patient_id="PT-024",
        vitals=dict(heart_rate=64, systolic_bp=108, diastolic_bp=70,
                    respiratory_rate=14, temperature_celsius=36.4,
                    oxygen_saturation=98.0, glasgow_coma_scale=15, pain_score=1),
        symptoms=["routine medication review", "mild fatigue", "occasional palpitations"],
        history=dict(age=72, sex="M", chief_complaint="palpitations and fatigue",
                     onset_hours=720.0,  # 1 month
                     past_medical_history=["heart failure (EF 30%)", "chronic AF", "CKD stage 3"],
                     current_medications=["digoxin 250mcg daily", "furosemide",
                                          "warfarin", "carvedilol"],
                     allergies=[], surgical_history=["ICD insertion 3 years ago"]),
        additional_context=(
            "ECG: scooped ST depression, short QT, regularized RR intervals. "
            "Digoxin level: 3.2 nmol/L (toxic range > 2.0). "
            "K+ 3.1 mEq/L. Creatinine 2.1 (baseline 1.6). "
            "Patient appears well and mobile, hence deceptively low initial triage concern."
        ),
        correct_triage_level=2,
        correct_actions=["hold digoxin immediately",
                         "continuous cardiac monitoring",
                         "IV potassium chloride replacement (electrolyte-guided)",
                         "DigiFab (digoxin-specific antibody fragments) assessment",
                         "cardiology consult",
                         "strict fluid balance (HF + CKD)",
                         "repeat ECG", "repeat digoxin level in 6 hours",
                         "review all interacting medications"],
        critical_flags=["digoxin toxicity - fatal arrhythmia risk",
                        "hypokalemia potentiating toxicity",
                        "AKI worsening digoxin clearance"],
        difficulty="hard",
        explanation=(
            "Digoxin toxicity with hypokalemia in AKI - classic lethal triad masked by "
            "patient appearing well. Regularized AF on ECG is pathognomonic. ESI 2."
        )
    ),

    PatientCase(
        patient_id="PT-025",
        vitals=dict(heart_rate=116, systolic_bp=106, diastolic_bp=68,
                    respiratory_rate=24, temperature_celsius=37.8,
                    oxygen_saturation=94.0, glasgow_coma_scale=14, pain_score=5),
        symptoms=["new-onset seizure (generalized)", "postictal confusion",
                  "mild fever", "headache preceding seizure"],
        history=dict(age=34, sex="F", chief_complaint="seizure",
                     onset_hours=0.5, past_medical_history=["HIV positive (CD4 80)",
                                                              "recent travel to sub-Saharan Africa"],
                     current_medications=["antiretroviral therapy (tenofovir/emtricitabine/dolutegravir)"],
                     allergies=[], surgical_history=[]),
        additional_context=(
            "No prior seizure history. CD4 80 - severely immunocompromised. "
            "CT head: ring-enhancing lesion right temporal lobe. "
            "Fundoscopy: no papilledema."
        ),
        correct_triage_level=1,
        correct_actions=["seizure precautions and safety",
                         "IV access x2", "IV benzodiazepine if seizing",
                         "CT head (already done) - MRI brain with contrast to follow",
                         "lumbar puncture (after MRI - ring enhancing lesion warrants caution)",
                         "toxoplasma serology, cryptococcal antigen",
                         "antiretroviral review", "infectious disease consult",
                         "empiric anti-toxoplasma therapy (pyrimethamine + sulfadiazine)",
                         "anti-epileptic medication",
                         "ICU monitoring"],
        critical_flags=["CNS opportunistic infection - cerebral toxoplasmosis likely",
                        "severe immunocompromise (CD4 80)",
                        "new seizure with ring-enhancing lesion",
                        "differential includes CNS lymphoma, cryptococcal meningitis"],
        difficulty="hard",
        explanation=(
            "Immunocompromised HIV patient with CD4 80, new seizure, ring-enhancing lesion. "
            "Cerebral toxoplasmosis vs CNS lymphoma. ESI 1."
        )
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Reward / Scoring Logic
# ─────────────────────────────────────────────────────────────────────────────

_CRITICAL_KEYWORDS = {
    "septic shock", "sepsis", "opioid", "overdose", "hemorrhage", "dissection",
    "stroke", "stemi", "nstemi", "acs", "heart block", "pe", "pulmonary embolism",
    "subarachnoid", "meningitis", "meningococcal", "ectopic", "toxoplasmosis",
    "arrhythmia", "digoxin toxic", "variceal", "shock", "respiratory failure",
    "airway", "intubation", "urosepsis", "cholangitis", "retinal artery",
    "cardiac arrest", "seizure", "cns", "status epilepticus",
}


def _normalize(text: str) -> str:
    return text.lower().strip()


def _flag_overlap(predicted: list[str], ground_truth: list[str]) -> float:
    """Compute recall-like overlap between critical flags."""
    if not ground_truth:
        return 1.0  # no flags expected - full credit if none raised
    if not predicted:
        return 0.0
    gt_tokens = set()
    for f in ground_truth:
        gt_tokens.update(_normalize(f).split())
    pred_tokens = set()
    for f in predicted:
        pred_tokens.update(_normalize(f).split())
    overlap = len(gt_tokens & pred_tokens)
    return min(1.0, overlap / max(len(gt_tokens), 1))


# Medical synonym groups — any word from a group matches any other
_SYN_GROUPS: list[set[str]] = [
    {"iv", "intravenous", "intravenously"},
    {"access", "line", "cannula", "catheter"},
    {"ecg", "ekg", "electrocardiogram", "electrocardiography"},
    {"ct", "ctscan", "computed", "tomography"},
    {"oxygen", "o2", "o\u2082", "supplemental"},
    {"antibiotic", "antibiotics", "antimicrobial"},
    {"consult", "consultation", "referral"},
    {"monitor", "monitoring", "surveillance"},
    {"fluid", "fluids", "resuscitation", "hydration"},
    {"blood", "haem", "hemo"},
    {"culture", "cultures"},
    {"naloxone", "narcan"},
    {"intubation", "intubate", "airway"},
    {"icu", "intensive", "critical"},
    {"cardiology", "cardiac", "cardiologist"},
    {"neurology", "neurologist"},
    {"surgery", "surgical", "surgeon"},
    {"cbc", "complete", "haemoglobin", "hemoglobin"},
    {"bmp", "metabolic", "electrolytes"},
    {"glucose", "sugar", "dextrose"},
    {"troponin", "troponins"},
    {"xray", "x-ray", "radiograph", "radiography"},
    {"ultrasound", "echo", "echocardiogram", "sonography"},
    {"npo", "nil", "fasting", "food"},
    {"analgesia", "pain", "analgetic", "analgesic"},
    {"urine", "urinalysis", "urea"},
    {"sepsis", "septic"},
    {"stroke", "tpa", "thrombolysis", "thrombolytic"},
    {"pacing", "pacemaker", "transcutaneous"},
    {"digoxin", "digoxin-specific", "digifab"},
    {"anticoagulation", "anticoagulant", "heparin", "lmwh", "doac"},
]

_STOPWORDS = frozenset({
    "and", "the", "or", "with", "of", "to", "a", "an", "if",
    "for", "in", "per", "vs", "x", "x2", "is", "be", "as",
    "at", "on", "by", "do", "not", "consider",
})


def _canonical_token(token: str) -> str:
    """Map a token to its synonym-group representative (the first element)."""
    for group in _SYN_GROUPS:
        if token in group:
            return min(group)  # stable representative
    return token


def _action_overlap(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Concept-aware action coverage score.

    Strategy:
    1. Extract meaningful words from both lists, dropping stopwords.
    2. Map each token through the synonym table so paraphrases match.
    3. Also generate bigrams to capture multi-word medical concepts.
    4. Score = |gt_concepts intersect pred_concepts| / |gt_concepts|.
    """
    if not ground_truth:
        return 1.0
    if not predicted:
        return 0.0

    def _tokenize_and_canonicalize(texts: list[str]) -> set[str]:
        tokens: set[str] = set()
        for text in texts:
            words = _normalize(text).replace("-", " ").replace("/", " ").split()
            cleaned = [w for w in words if w not in _STOPWORDS and len(w) > 1]
            # unigrams
            for w in cleaned:
                tokens.add(_canonical_token(w))
            # bigrams (capture phrases like "blood cultures", "cardiac monitoring")
            for i in range(len(cleaned) - 1):
                bigram = _canonical_token(cleaned[i]) + "_" + _canonical_token(cleaned[i + 1])
                tokens.add(bigram)
        return tokens

    gt_concepts = _tokenize_and_canonicalize(ground_truth)
    pred_concepts = _tokenize_and_canonicalize(predicted)

    if not gt_concepts:
        return 1.0

    matched = gt_concepts & pred_concepts
    return min(1.0, len(matched) / len(gt_concepts))


def compute_reward(action: Action, case: PatientCase) -> Reward:
    """
    Score an agent's action against the ground-truth patient case.

    Scoring components (all normalized to [0, 1]):
    - triage_score    : 1.0 exact, 0.5 for +-1 level, 0.0 for >=2 level error
    - action_score    : keyword overlap with canonical actions
    - flag_recall     : proportion of critical flags caught
    - flag_penalty    : penalty for missing critical flags
    - overtriage_pen  : penalty for over-triaging stable patients
    """
    breakdown: dict[str, float] = {}

    # 1. Triage level accuracy
    level_diff = abs(action.triage_level - case.correct_triage_level)
    if level_diff == 0:
        triage_score = 1.0
    elif level_diff == 1:
        triage_score = 0.5
    else:
        triage_score = 0.0
    breakdown["triage_level_score"] = triage_score

    # 2. Recommended actions coverage
    action_score = _action_overlap(action.recommended_actions, case.correct_actions)
    breakdown["action_coverage_score"] = round(action_score, 4)

    # 3. Critical flag recall
    flag_recall = _flag_overlap(action.critical_flags, case.critical_flags)
    breakdown["flag_recall_score"] = round(flag_recall, 4)

    # 4. Penalty: missing life-threatening flags (case has flags, agent misses them)
    missing_flag_penalty = 0.0
    if case.critical_flags and not action.critical_flags:
        missing_flag_penalty = 0.3
    elif case.critical_flags:
        gt_tokens = set()
        for f in case.critical_flags:
            gt_tokens.update(_normalize(f).split())
        pred_tokens = set()
        for f in action.critical_flags:
            pred_tokens.update(_normalize(f).split())
        missed_fraction = 1.0 - (len(gt_tokens & pred_tokens) / len(gt_tokens))
        missing_flag_penalty = round(0.3 * missed_fraction, 4)
    breakdown["missing_flag_penalty"] = -missing_flag_penalty

    # 5. Penalty: over-triaging stable patients (assigned ESI 1 or 2 to ESI 4 or 5 cases)
    overtriage_penalty = 0.0
    if case.correct_triage_level >= 4 and action.triage_level <= 2:
        overtriage_penalty = 0.1
    breakdown["overtriage_penalty"] = -overtriage_penalty

    # 6. Bonus: catching any critical flag when case has them (encourages vigilance)
    flag_bonus = 0.0
    if case.critical_flags and action.critical_flags:
        flag_bonus = round(0.10 * flag_recall, 4)
    breakdown["critical_flag_bonus"] = flag_bonus

    # Weighted sum
    # Weights: triage accuracy 50% (most clinically important),
    #          action coverage 20% (paraphrase-tolerant),
    #          flag recall 20% (life-threatening vigilance),
    #          flag bonus up to +10%, penalties as above
    raw = (
        0.50 * triage_score
        + 0.20 * action_score
        + 0.20 * flag_recall
        + flag_bonus
        - missing_flag_penalty
        - overtriage_penalty
    )
    score = max(0.0, min(1.0, raw))
    breakdown["total"] = round(score, 4)

    # Build feedback string
    feedback_parts = []
    if triage_score == 1.0:
        feedback_parts.append("[OK] Triage level correct.")
    elif triage_score == 0.5:
        feedback_parts.append(
            f"[WARN] Triage level +-1 (assigned {action.triage_level}, correct {case.correct_triage_level})."
        )
    else:
        feedback_parts.append(
            f"[FAIL] Triage level significantly off (assigned {action.triage_level}, correct {case.correct_triage_level})."
        )
    if missing_flag_penalty > 0:
        feedback_parts.append(
            f"[FAIL] Critical flags missed: {case.critical_flags}"
        )
    if overtriage_penalty > 0:
        feedback_parts.append("[WARN] Over-triage detected for a stable patient.")
    feedback_parts.append(f"Clinical note: {case.explanation}")

    return Reward(
        score=round(score, 4),
        breakdown=breakdown,
        feedback=" | ".join(feedback_parts)
    )


# ─────────────────────────────────────────────────────────────────────────────
# MedicalTriageEnv
# ─────────────────────────────────────────────────────────────────────────────

class MedicalTriageEnv:
    """
    OpenEnv-compatible Medical Triage & Clinical Decision Support environment.

    Episode flow:
        obs = env.reset(case_ids=[...])   # start new episode
        while not done:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._all_cases: list[PatientCase] = PATIENT_CASES[:]
        self._episode_cases: list[PatientCase] = []
        self._current_index: int = 0
        self._episode_rewards: list[float] = []
        self._last_action: Optional[Action] = None
        self._last_reward: Optional[Reward] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(
        self,
        case_ids: Optional[list[str]] = None,
        shuffle: bool = True,
    ) -> Observation:
        """
        Start a new episode.

        Args:
            case_ids: Specific patient IDs to include. If None, all cases are used.
            shuffle:  Randomise case order within the episode.

        Returns:
            First Observation in the episode.
        """
        if case_ids is not None:
            id_set = set(case_ids)
            self._episode_cases = [c for c in self._all_cases if c.patient_id in id_set]
            if not self._episode_cases:
                raise ValueError(f"No matching cases found for ids: {case_ids}")
        else:
            self._episode_cases = self._all_cases[:]

        if shuffle:
            self._rng.shuffle(self._episode_cases)

        self._current_index = 0
        self._episode_rewards = []
        self._last_action = None
        self._last_reward = None

        return self._make_observation()

    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict]:
        """
        Process the agent's action for the current patient.

        Args:
            action: Agent's triage decision.

        Returns:
            (next_observation, reward, done, info)
            - next_observation is None when episode is done.
        """
        if not self._episode_cases:
            raise RuntimeError("Call reset() before step().")
        if self._current_index >= len(self._episode_cases):
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        current_case = self._episode_cases[self._current_index]
        reward = compute_reward(action, current_case)

        self._last_action = action
        self._last_reward = reward
        self._episode_rewards.append(reward.score)

        self._current_index += 1
        done = self._current_index >= len(self._episode_cases)

        next_obs = None if done else self._make_observation()

        info = {
            "case_id": current_case.patient_id,
            "correct_triage_level": current_case.correct_triage_level,
            "correct_actions": current_case.correct_actions,
            "correct_critical_flags": current_case.critical_flags,
            "difficulty": current_case.difficulty,
            "explanation": current_case.explanation,
            "reward_breakdown": reward.breakdown,
            "episode_scores_so_far": list(self._episode_rewards),
            "mean_score_so_far": round(
                sum(self._episode_rewards) / len(self._episode_rewards), 4
            ),
        }

        return next_obs, reward, done, info

    def state(self) -> dict[str, Any]:
        """Return full current environment state as a dictionary."""
        return {
            "episode_length": len(self._episode_cases),
            "current_index": self._current_index,
            "done": self._current_index >= len(self._episode_cases),
            "episode_scores": list(self._episode_rewards),
            "mean_score": (
                round(sum(self._episode_rewards) / len(self._episode_rewards), 4)
                if self._episode_rewards else None
            ),
            "current_patient_id": (
                self._episode_cases[self._current_index].patient_id
                if self._current_index < len(self._episode_cases)
                else None
            ),
            "last_action": self._last_action.model_dump() if self._last_action else None,
            "last_reward": self._last_reward.model_dump() if self._last_reward else None,
            "all_case_ids": [c.patient_id for c in self._episode_cases],
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        case = self._episode_cases[self._current_index]
        return Observation(
            patient_id=case.patient_id,
            vitals=Vitals(**case.vitals),
            symptoms=case.symptoms,
            history=PatientHistory(**case.history),
            additional_context=case.additional_context,
            timestamp=datetime.utcnow().isoformat() + "Z",
            case_index=self._current_index,
            total_cases=len(self._episode_cases),
        )

    # ── Utility ────────────────────────────────────────────────────────────

    @property
    def case_ids_by_difficulty(self) -> dict[str, list[str]]:
        """Return case IDs grouped by difficulty."""
        result: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
        for c in self._all_cases:
            result.setdefault(c.difficulty, []).append(c.patient_id)
        return result

    def get_case_by_id(self, patient_id: str) -> Optional[PatientCase]:
        for c in self._all_cases:
            if c.patient_id == patient_id:
                return c
        return None
