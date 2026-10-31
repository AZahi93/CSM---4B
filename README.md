# TravelGenie - Multi-Agent Travel Planner

Stack: **Groq** (Llama 3.3 70B + Llama 3.1 8B) · **LangGraph + LangChain** · **FastAPI** (SSE streaming) · **React + Vite**

## Run it
Backend (Python 3.10+):
```
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # paste your free key from console.groq.com
uvicorn main:app --reload --port 8000
```
Frontend (Node 18+), in a second terminal:
```
cd frontend
npm install
npm run dev                                           # open http://localhost:5173
```

## Agents (graph.py)
| Agent | What it does | Uses |
|---|---|---|
| Orchestrator | Algorithm A: splits request into tasks, routes to agents | Groq (fast) |
| Preference | Extracts interests, pace, diet; geocodes places | Groq (fast), Open-Meteo geocoding |
| Transport | Algorithm C: weighted ranking of flights/train/bus | simulated prices |
| Stay | Algorithm C: weighted ranking (price, rating, distance) | simulated hotels |
| Activities | ReAct agent (reason -> tool -> observe) | Wikipedia geosearch, Overpass (OpenStreetMap) |
| Weather | Live forecast + packing summary | Open-Meteo |
| Budget | Picks best-ranked combination within budget | pure Python |
| Scheduler | Algorithm D: nearest-neighbour route, then timed plan | Groq (main) |
| Critic | Rule checks + LLM review; loops back to Scheduler (max 3 runs) | Groq (fast) |

Flow: orchestrator -> preference -> [transport | stay | activities | weather] in parallel -> budget -> scheduler -> critic -> (issues? back to scheduler).

## Honest limits
- Transport and hotel data is **simulated** (clearly labelled in the UI). For real data, replace `flight_agent`/`hotel_agent`
  with Amadeus Self-Service API calls (free test tier).
- Weather forecast covers the next 16 days only.
- Groq free tier has rate limits; if you hit them, set `GROQ_MODEL` to `llama-3.1-8b-instant`.
