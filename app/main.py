from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import services

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Road Accident Risk Prediction API")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

ADVICE = {
    "Low": "Conditions look normal. Drive carefully and follow speed limits.",
    "Medium": "Elevated risk. Reduce speed, keep a safe distance and stay alert.",
    "High": "High risk. Avoid unnecessary travel here if possible; slow down, use headlights, stay extra alert.",
}


def level(p: float) -> str:
    return "High" if p >= 0.5 else "Medium" if p >= 0.25 else "Low"


def predict_df(df: pd.DataFrame) -> np.ndarray:
    if model is None:
        raise HTTPException(503, "Model not trained. Run: python src/train.py")
    if "speed_limit" not in df:
        df["speed_limit"] = df["road_type"].map(services.SPEED)
    return model.predict_proba(df)[:, 1]


def risk_factors(row: dict, base: float) -> list:
    """What-if analysis: how much would risk drop if each bad condition were removed?"""
    tests = [
        ("Rain", row["rainfall_mm"] > 1, {"rainfall_mm": 0.0}),
        ("Low visibility", row["visibility_km"] < 5, {"visibility_km": 10.0}),
        ("Night-time", row["hour"] >= 22 or row["hour"] <= 5, {"hour": 12}),
        ("Junction nearby", row["junction"] == 1, {"junction": 0}),
        ("Heavy traffic", row["traffic_volume"] > 2500, {"traffic_volume": 1000}),
    ]
    active = [(n, ch) for n, cond, ch in tests if cond]
    if not active:
        return []
    variants = pd.DataFrame([{**row, **ch} for _, ch in active])
    probs = predict_df(variants)
    out = [{"factor": n, "impact_pct": round((base - float(p)) * 100, 1)} for (n, _), p in zip(active, probs)]
    return sorted([o for o in out if o["impact_pct"] > 0.1], key=lambda o: -o["impact_pct"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/analyze")
def analyze(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
            hour: Optional[int] = Query(None, ge=0, le=23)):
    """Give a latitude/longitude and get weather, road info, time, traffic estimate and accident risk."""
    now = services.now_ist()
    h = now.hour if hour is None else hour
    weather = services.get_weather(lat, lon)
    road = services.get_road_info(lat, lon)
    traffic = services.estimate_traffic(h, road["road_type"])

    row = {"lat": lat, "lon": lon, "hour": h, "day_of_week": now.weekday(), "month": now.month,
           "rainfall_mm": weather["rainfall_mm"], "visibility_km": weather["visibility_km"],
           "temperature_c": weather["temperature_c"], "road_type": road["road_type"],
           "junction": road["junction"], "traffic_volume": traffic, "speed_limit": road["speed_limit"]}
    p = float(predict_df(pd.DataFrame([row]))[0])
    lvl = level(p)
    return {
        "location": {"lat": lat, "lon": lon, "address": services.reverse_geocode(lat, lon)},
        "time": {"datetime_ist": now.strftime("%Y-%m-%d %H:%M"), "hour_used": h,
                 "day": now.strftime("%A")},
        "weather": weather,
        "road": road,
        "traffic_volume_estimate": traffic,
        "risk": {"probability": round(p, 4), "level": lvl, "advice": ADVICE[lvl],
                 "factors": risk_factors(row, p)},
    }


@app.get("/risk-grid")
def risk_grid(hour: int = 18, rainfall_mm: float = 0, visibility_km: float = 8, n: int = 20,
              lat_min: float = 28.35, lat_max: float = 28.85, lon_min: float = 76.95, lon_max: float = 77.55):
    """Risk for a grid of points under given conditions (map overlay)."""
    n = min(max(n, 5), 30)
    lats, lons = np.linspace(lat_min, lat_max, n), np.linspace(lon_min, lon_max, n)
    g = pd.DataFrame([(a, b) for a in lats for b in lons], columns=["lat", "lon"])
    g = g.assign(hour=hour, day_of_week=2, month=10, rainfall_mm=rainfall_mm, visibility_km=visibility_km,
                 temperature_c=28, road_type="urban", junction=0, traffic_volume=2000)
    g["risk"] = predict_df(g).round(4)
    return g[["lat", "lon", "risk"]].to_dict(orient="records")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
