import pytest
from src.geocode import geocode_place, geocode_first


class _FakeLoc:
    def __init__(self, address, lat, lon):
        self.address = address
        self.latitude = lat
        self.longitude = lon
        self.raw = {"boundingbox": ["25.0", "26.0", "85.0", "86.0"], "type": "city", "class": "place"}


class _FakeGeo:
    def __init__(self, *a, **k):
        pass

    def geocode(self, q, **k):
        if q == "Nowhere XYZ":
            return []
        return [_FakeLoc("Patna, Bihar, India", 25.5941, 85.1376)]


def test_geocode_success(monkeypatch):
    # Nominatim is imported inside geocode_place(); patch it at its source.
    import geopy.geocoders
    monkeypatch.setattr(geopy.geocoders, "Nominatim", _FakeGeo)
    res = geocode_place("Patna")
    assert len(res) == 1
    assert res[0]["latitude"] == pytest.approx(25.5941)
    assert geocode_first("Patna")["longitude"] == pytest.approx(85.1376)


def test_geocode_not_found(monkeypatch):
    import geopy.geocoders
    monkeypatch.setattr(geopy.geocoders, "Nominatim", _FakeGeo)
    assert geocode_place("Nowhere XYZ") == []
    assert geocode_first("Nowhere XYZ") is None


def test_geocode_empty_query():
    with pytest.raises(ValueError):
        geocode_place("   ")
