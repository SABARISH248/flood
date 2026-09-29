"""Place-search (geocoding) endpoints."""
from fastapi import APIRouter, HTTPException
from src.geocode import geocode_place

router = APIRouter(prefix="/geocode", tags=["geocode"])

@router.get("")
def geocode(place: str, limit: int = 5):
    try:
        results = geocode_place(place, limit=limit)
    except ImportError as e:
        raise HTTPException(500, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    if not results:
        raise HTTPException(404, f"No matching place found for '{place}'")
    return {"query": place, "results": results}
