"""External data tools. Weather, geocoding, attractions and restaurants use free,
key-less public APIs. Flights and hotels are SIMULATED (see graph.py) because real
inventory APIs (Amadeus, Booking, etc.) need paid/approved keys."""
import math
import httpx
from langchain_core.tools import tool

UA = {"User-Agent": "TravelGenie-college-project/1.0"}


def geocode(place: str):
    r = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                  params={"name": place, "count": 1}, timeout=15).json()
    res = r.get("results")
    if not res:
        return None
    x = res[0]
    return {"name": x["name"], "lat": x["latitude"], "lon": x["longitude"],
            "country": x.get("country", "")}


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def get_forecast(lat: float, lon: float, start: str, end: str):
    d = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=20, params={
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 16,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"}).json()["daily"]
    rows = [{"date": t, "tmax": mx, "tmin": mn, "rain_mm": rn}
            for t, mx, mn, rn in zip(d["time"], d["temperature_2m_max"],
                                     d["temperature_2m_min"], d["precipitation_sum"])]
    sel = [r for r in rows if start <= r["date"] <= end]
    note = None
    if not sel:
        sel, note = rows[:7], "Trip dates are beyond the 16-day forecast window; showing the upcoming outlook instead."
    return {"days": sel, "note": note}


@tool
def search_attractions(lat: float, lon: float) -> list:
    """Find notable landmarks and attractions within 10 km of the given coordinates."""
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
            "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
            "gsradius": 10000, "gslimit": 30, "format": "json"}).json()
        return [{"name": x["title"], "lat": x["lat"], "lon": x["lon"]}
                for x in r["query"]["geosearch"]]
    except Exception:
        return []


@tool
def search_restaurants(lat: float, lon: float) -> list:
    """Find restaurants within 5 km of the given coordinates, with cuisine when known."""
    q = f'[out:json][timeout:20];node["amenity"="restaurant"]["name"](around:5000,{lat},{lon});out 30;'
    try:
        r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": q},
                       headers=UA, timeout=25).json()
        return [{"name": e["tags"]["name"], "cuisine": e["tags"].get("cuisine", "unknown"),
                 "lat": e["lat"], "lon": e["lon"]} for e in r.get("elements", [])]
    except Exception:
        return []
