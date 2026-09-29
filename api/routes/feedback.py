from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from src.feedback import submit_field_report, get_all_field_reports, verify_field_report

router = APIRouter(prefix="/feedback", tags=["Field Feedback"])


class FeedbackSubmission(BaseModel):
    latitude: float = Field(..., example=25.5941)
    longitude: float = Field(..., example=85.1376)
    date: str = Field(..., example="2026-09-28")
    depth_m: float = Field(..., example=1.5)
    description: str = Field(..., example="Submerged road and farmland")
    reporter_name: Optional[str] = "Anonymous"
    photo_path: Optional[str] = ""


class VerificationRequest(BaseModel):
    record_id: int
    verified: bool = True
    verification_notes: Optional[str] = "Verified by domain expert"


@router.post("")
def submit_feedback(req: FeedbackSubmission):
    try:
        rec_id = submit_field_report(
            lat=req.latitude,
            lon=req.longitude,
            date_str=req.date,
            depth_m=req.depth_m,
            description=req.description,
            reporter_name=req.reporter_name or "Anonymous",
            photo_path=req.photo_path
        )
        return {"status": "success", "record_id": rec_id, "message": "Field report submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
def list_feedback_records():
    try:
        df = get_all_field_reports()
        return {"count": len(df), "records": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify")
def verify_feedback(req: VerificationRequest):
    try:
        verify_field_report(req.record_id, req.verified, req.verification_notes or "")
        return {"status": "success", "record_id": req.record_id, "verified": req.verified}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
