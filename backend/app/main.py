from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import uuid, shutil, re
from .ml_pipeline import get_model_and_metrics, predict_relationship

app = FastAPI(title="ClaimLens Evidence Analysis API", version="1.0.0",
              description="Research prototype for uncertainty-aware insurance claim evidence review.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
DATA = Path(__file__).resolve().parents[1] / "data" / "uploads"
DATA.mkdir(parents=True, exist_ok=True)
CLAIMS = {}

class ClaimInput(BaseModel):
    claim_number: str = Field(min_length=3, max_length=40)
    policy_number: str = Field(min_length=3, max_length=40)
    incident_date: str
    vehicle_registration: str
    damage_type: str
    claimed_expenses: float = Field(ge=0)
    claim_type: str = "Motor Vehicle"

class RelationshipInput(BaseModel):
    claim_detail: str
    evidence_text: str

@app.get("/")
def root():
    return {"name":"ClaimLens API","status":"running","docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok","service":"ClaimLens"}

@app.get("/api/metrics")
def metrics():
    _, result = get_model_and_metrics()
    return result

@app.post("/api/predict-relationship")
def predict(data: RelationshipInput):
    return predict_relationship(data.claim_detail, data.evidence_text)

@app.get("/api/claims")
def get_claims():
    if not CLAIMS:
        return {"claims": demo_claims()}
    return {"claims": list(CLAIMS.values()) + demo_claims()}

def demo_claims():
    return [
      {"id":"CL-45821","claim_number":"CL-45821","date":"12 Apr 2025","type":"Motor Vehicle","status":"Under Review","priority":"High","supported":2,"contradicted":1,"insufficient":1},
      {"id":"CL-45820","claim_number":"CL-45820","date":"10 Apr 2025","type":"Travel","status":"Inconsistencies Found","priority":"High","supported":1,"contradicted":2,"insufficient":0},
      {"id":"CL-45819","claim_number":"CL-45819","date":"09 Apr 2025","type":"Health","status":"Evidence Insufficient","priority":"Medium","supported":1,"contradicted":0,"insufficient":2},
      {"id":"CL-45818","claim_number":"CL-45818","date":"08 Apr 2025","type":"Motor Vehicle","status":"Resolved","priority":"Low","supported":3,"contradicted":0,"insufficient":0},
      {"id":"CL-45817","claim_number":"CL-45817","date":"07 Apr 2025","type":"Travel","status":"Under Review","priority":"Medium","supported":2,"contradicted":1,"insufficient":1},
    ]

@app.post("/api/claims")
def create_claim(claim: ClaimInput):
    claim_id = claim.claim_number.strip().upper()
    record = {"id":claim_id, "claim_number":claim_id, "date":claim.incident_date,
      "type":claim.claim_type, "status":"Under Review", "priority":"Medium",
      "supported":0,"contradicted":0,"insufficient":1,
      "policy_number":claim.policy_number, "vehicle_registration":claim.vehicle_registration,
      "damage_type":claim.damage_type, "claimed_expenses":claim.claimed_expenses}
    CLAIMS[claim_id] = record
    return {"message":"Claim created", "claim":record}

@app.post("/api/claims/{claim_id}/evidence")
async def upload_evidence(claim_id: str, files: list[UploadFile] = File(...)):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", claim_id)
    folder = DATA / safe_id
    folder.mkdir(parents=True, exist_ok=True)
    accepted = []
    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in {".pdf",".png",".jpg",".jpeg",".doc",".docx",".txt"}:
            continue
        target = folder / f"{uuid.uuid4().hex}{ext}"
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        accepted.append({"name":upload.filename,"size_bytes":target.stat().st_size,"type":ext})
    return {"claim_id":claim_id,"uploaded":accepted,"count":len(accepted),
            "note":"Files are stored locally. This prototype does not claim full OCR/image analysis."}

@app.get("/api/claims/{claim_id}/analysis")
def claim_analysis(claim_id: str):
    # Demonstration evidence relationships; clearly marked as demo rather than extracted real facts.
    return {
      "claim_id":claim_id,
      "demo_mode":True,
      "findings":[
        {"id":"F-01","detail":"Front bumper damage","sources":["Accident Report","Vehicle Photo 1"],"relationship":"Supported","confidence":0.94,"explanation":"The report describes front bumper damage and the supplied example photograph visibly shows the front bumper. Confirm against the original full-resolution files."},
        {"id":"F-02","detail":"Repair component","sources":["Repair Invoice","Vehicle Photo 1"],"relationship":"Contradicted","confidence":0.82,"explanation":"The example invoice lists a rear bumper replacement while the reviewed example photograph shows front bumper damage. These may refer to different repairs; verify before drawing conclusions."},
        {"id":"F-03","detail":"Incident date","sources":["Accident Report","Hospital Record"],"relationship":"Supported","confidence":0.77,"explanation":"The example records contain the same incident date."},
        {"id":"F-04","detail":"Rear quarter-panel condition","sources":["Vehicle Photo 1"],"relationship":"Insufficient evidence","confidence":0.91,"explanation":"The available example image does not provide a clear view of the rear quarter panel. Request a focused photograph."}
      ],
      "recommended_actions":[
        "Request a clear photograph of the rear quarter panel.",
        "Confirm the invoice repair component with the repair workshop.",
        "Verify dates using the original, dated source records."
      ],
      "disclaimer":"Demonstration findings only. No finding is a fraud verdict."
    }
