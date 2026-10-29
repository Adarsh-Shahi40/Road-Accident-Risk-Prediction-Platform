"""Creates SYNTHETIC accident data so the project runs end to end.
Replace data/accidents.csv with real data (same columns) for real use."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
N = 20000

# Fake accident hotspots around Delhi NCR (lat, lon)
HOTSPOTS = np.array([[28.61, 77.21], [28.70, 77.10], [28.46, 77.03],
                     [28.54, 77.39], [28.67, 77.45]])
ROAD_TYPES = ["national_highway", "state_highway", "urban", "rural"]
SPEED = {"national_highway": 90, "state_highway": 70, "urban": 40, "rural": 50}
ROAD_EFFECT = {"national_highway": 0.6, "state_highway": 0.3, "urban": 0.1, "rural": 0.4}


def main():
    df = pd.DataFrame({
        "lat": rng.uniform(28.35, 28.85, N),
        "lon": rng.uniform(76.95, 77.55, N),
        "hour": rng.integers(0, 24, N),
        "day_of_week": rng.integers(0, 7, N),
        "month": rng.integers(1, 13, N),
        "rainfall_mm": rng.gamma(0.5, 4, N).round(1),
        "visibility_km": np.clip(rng.normal(6, 2.5, N), 0.1, 10).round(1),
        "temperature_c": rng.normal(28, 8, N).round(1),
        "road_type": rng.choice(ROAD_TYPES, N, p=[.25, .25, .35, .15]),
        "junction": rng.integers(0, 2, N),
        "traffic_volume": rng.integers(100, 5000, N),
    })
    df["speed_limit"] = df["road_type"].map(SPEED)

    # distance to nearest hotspot
    d = np.min(np.linalg.norm(
        df[["lat", "lon"]].values[:, None, :] - HOTSPOTS[None, :, :], axis=2), axis=1)
    night = ((df.hour >= 22) | (df.hour <= 5)).astype(int)
    rush = df.hour.isin([8, 9, 17, 18, 19]).astype(int)

    logit = (-3.6 + 0.9 * night + 0.4 * rush + 0.06 * df.rainfall_mm
             + 0.15 * (6 - df.visibility_km) + 0.5 * df.junction
             + 0.8 * df.traffic_volume / 5000 + 0.8 * df.speed_limit / 100
             + df.road_type.map(ROAD_EFFECT) + 1.5 * np.exp(-d / 0.05))
    p = 1 / (1 + np.exp(-logit))
    df["accident"] = (rng.random(N) < p).astype(int)

    out = ROOT / "data" / "accidents.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out} | accident rate: {df.accident.mean():.2%}")


if __name__ == "__main__":
    main()
