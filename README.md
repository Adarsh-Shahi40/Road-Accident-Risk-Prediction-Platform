# Road Accident Risk Prediction Platform

Predicts the probability of a road accident for a location and conditions (time, weather,
road type, traffic) and shows risk on an interactive map.

**Stack:** Python, scikit-learn, FastAPI, Leaflet (OpenStreetMap).

## Project structure
```
road-accident-risk/
├── app/
│   ├── main.py            # FastAPI backend
│   └── static/index.html  # Map frontend
├── src/
│   ├── generate_data.py   # Creates synthetic demo data
│   └── train.py           # Trains and saves the model
├── data/                  # accidents.csv goes here
├── models/                # risk_model.joblib is saved here
└── requirements.txt
```

## Setup (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py     # demo data 
python src/train.py             # train the model
uvicorn app.main:app --reload   # start the server
```
Open http://127.0.0.1:8000, press **Use my live location** (or click the map / type lat,long).
The app then fetches weather (Open-Meteo), road type/speed/junction (OpenStreetMap), address and
current time automatically, and predicts the risk. Needs internet access. API docs at http://127.0.0.1:8000/docs).

## Interface features
Speedometer-style risk gauge, place search, city shortcuts, live-location follow mode,
risk heat map for the visible area, recent checks and a shareable link (`?lat=..&lon=..`).

## How it works
`GET /analyze?lat=..&lon=..` -> `app/services.py` gathers weather, road, address, time and estimated
traffic -> model predicts risk -> what-if analysis lists which factors raise it.
Geolocation in browsers works on localhost or HTTPS only.

## Using real data
Replace `data/accidents.csv` with your own file using the same columns
(`lat, lon, hour, day_of_week, month, rainfall_mm, visibility_km, temperature_c,
road_type, junction, traffic_volume, speed_limit, accident`), then re-run `src/train.py`.
Note: real accident data only contains accidents, so you must also build "no accident"
samples (same segments and times without a crash) to get the `accident = 0` rows.

