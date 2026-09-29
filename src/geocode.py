"""Place search via Nominatim (OpenStreetMap). Free, no API key; usage limits apply."""
from typing import List, Dict, Any, Optional

USER_AGENT = "FloodSense-India/1.0 (flood-risk-dashboard)"


def geocode_place(query: str, country_codes: str = "in", limit: int = 5,
                  try_global_fallback: bool = True) -> List[Dict[str, Any]]:
    """
    Geocode a place name to coordinates.

    Parameters
    ----------
    query : str
        Place name, e.g. "Patna", "Guwahati, Assam".
    country_codes : str
        Bias to country (default "in" for India). Set to None for global.
    limit : int
        Max results.
    try_global_fallback : bool
        If biased search finds nothing, retry without country bias.

    Returns
    -------
    list of {display_name, latitude, longitude, bbox, type}
    Empty list if nothing found. Raises on bad input / missing geopy / service error.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    except ImportError:
        raise ImportError("geopy not installed. Run: pip install geopy")

    if not query or not query.strip():
        raise ValueError("Empty place query")

    geo = Nominatim(user_agent=USER_AGENT, timeout=10)
    q = query.strip()

    def _search(cc):
        kwargs = {"exactly_one": False, "limit": limit, "addressdetails": True}
        if cc:
            kwargs["country_codes"] = cc
        return geo.geocode(q, **kwargs)

    try:
        locs = _search(country_codes)
        if not locs and try_global_fallback and country_codes:
            locs = _search(None)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise ConnectionError(f"Geocoding service unavailable: {e}")

    if not locs:
        return []

    results = []
    for loc in locs:
        results.append({
            "display_name": loc.address,
            "latitude": float(loc.latitude),
            "longitude": float(loc.longitude),
            "bbox": loc.raw.get("boundingbox"),
            "type": loc.raw.get("type"),
            "category": loc.raw.get("class"),
        })
    return results


def geocode_first(query: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Return the top geocoding result, or None if nothing found."""
    results = geocode_place(query, **kwargs)
    return results[0] if results else None
