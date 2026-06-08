# ✈️ Satisfaction Predictor — FastAPI Deployment

## Quick Start

### Local (Python)
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Visit: http://localhost:8000/docs
```

### Docker
```bash
docker build -t satisfaction-api .
docker run -p 8000:8000 satisfaction-api
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Liveness probe |
| POST | `/predict` | Single passenger prediction |
| POST | `/predict/batch` | Batch predictions (up to 1,000) |
| GET | `/model/info` | Model metadata & performance |

## Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_type": "loyal",
    "age": 42,
    "travel_type": "business",
    "flight_class": "Business",
    "flight_distance": 1500,
    "seat_comfort": 5,
    "timing_convenience": 4,
    "food_and_drink": 3,
    "gate_location": 3,
    "inflight_wifi": 4,
    "inflight_entertainment": 5,
    "online_support": 4,
    "ease_of_booking": 4,
    "onboard_service": 5,
    "leg_room": 4,
    "baggage_handling": 4,
    "checkin_service": 4,
    "cleanliness": 5,
    "online_boarding": 4,
    "departure_delay": 0,
    "arrival_delay": 0
  }'
```

## Example Response
```json
{
  "satisfaction_probability": 0.9921,
  "predicted_class": "satisfied",
  "confidence": "Very High",
  "risk_tier": "Low Risk",
  "recommendation": "Passenger is very likely satisfied. Include in NPS champion outreach."
}
```
