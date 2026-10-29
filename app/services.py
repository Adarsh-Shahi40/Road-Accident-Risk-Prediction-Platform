"""Fetches everything the model needs from just a latitude/longitude.
Free APIs, no keys needed: Open-Meteo (weather), OpenStreetMap Overpass (road), Nominatim (address)."""
import re
from datetime import datetime, timedelta, timezone

import requests

HEADERS = {"User-Agent": "road-accident-risk-demo/1.0 (learning project)"}
TIMEOUT = 12
IST = timezone(timedelta(hours=5, minutes=30))

SPEED = {"national_highway": 90, "state_highway": 70, "urban": 40, "rural": 50}
# OSM 'highway' tag -> our road types (in India primary roads are mostly NH)
HIGHWAY_TO_TYPE = {
    "motorway": "national_highway", "trunk": "national_highway", "primary": "national_highway",
    "secondary": "state_highway",
    "tertiary": "urban", "unclassified": "urban", "residential": "urban",
    "living_street": "urban", "service": "urban",
    "track": "rural", "path": "rural",
}
PRIORITY = list(HIGHWAY_TO_TYPE)  # most important road first


def now_ist() -> datetime:
    return datetime.now(IST)


def get_weather(lat: float, lon: float) -> dict:
    default = {"temperature_c": 28.0, "rainfall_mm": 0.0, "visibility_km": 8.0, "source": "default (weather API unavailable)"}
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation",
            "hourly": "visibility", "forecast_days": 1, "timezone": "auto"}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        cur = d["current"]
        key = cur["time"][:13] + ":00"
        times = d["hourly"]["time"]
        vis_m = d["hourly"]["visibility"][times.index(key)] if key in times else None
        vis_km = 8.0 if vis_m is None else round(min(vis_m / 1000, 10), 1)
        return {"temperature_c": cur["temperature_2m"], "rainfall_mm": cur["precipitation"],
                "visibility_km": vis_km, "source": "Open-Meteo"}
    except Exception:
        return default


def parse_road(elements: list) -> dict | None:
    ways = [e for e in elements if e.get("type") == "way" and "highway" in e.get("tags", {})]
    if not ways:
        return None
    best = min(ways, key=lambda w: PRIORITY.index(w["tags"]["highway"]) if w["tags"]["highway"] in PRIORITY else 99)
    tags = best["tags"]
    road_type = HIGHWAY_TO_TYPE.get(tags["highway"], "urban")
    m = re.match(r"\d+", str(tags.get("maxspeed", "")))
    signals = any(e.get("tags", {}).get("highway") == "traffic_signals" for e in elements if e.get("type") == "node")
    return {
        "road_name": tags.get("name") or tags.get("ref") or "Unnamed road",
        "osm_highway_type": tags["highway"],
        "road_type": road_type,
        "speed_limit": int(m.group()) if m else SPEED[road_type],
        "junction": int(signals or len({w["id"] for w in ways}) >= 3),
        "source": "OpenStreetMap",
    }


def get_road_info(lat: float, lon: float) -> dict:
    default = {"road_name": "Unknown", "osm_highway_type": "unknown", "road_type": "urban",
               "speed_limit": 40, "junction": 0, "source": "default (no road found / API unavailable)"}
    q = (f'[out:json][timeout:10];(way(around:40,{lat},{lon})["highway"];'
         f'node(around:40,{lat},{lon})["highway"="traffic_signals"];);out tags;')
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return parse_road(r.json().get("elements", [])) or default
    except Exception:
        return default


def reverse_geocode(lat: float, lon: float) -> str | None:
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", headers=HEADERS, timeout=TIMEOUT,
                         params={"format": "jsonv2", "lat": lat, "lon": lon})
        r.raise_for_status()
        return r.json().get("display_name")
    except Exception:
        return None


def estimate_traffic(hour: int, road_type: str) -> int:
    """Rough heuristic (vehicles/hour). Replace with a real traffic API (TomTom/HERE) later."""
    base = {"national_highway": 3000, "state_highway": 2200, "urban": 3500, "rural": 800}[road_type]
    f = 0.3 if hour <= 5 else 1.3 if hour in (8, 9, 10, 17, 18, 19, 20) else 0.8
    return int(base * f)
