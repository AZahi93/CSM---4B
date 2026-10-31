"""TravelGenie multi-agent graph (LangGraph + Groq).

orchestrator -> preference -> [flight | hotel | activities | weather] (parallel)
             -> budget -> scheduler -> critic --(issues? loop)--> scheduler
"""
import json, math, os, random, re
from datetime import date
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import geocode, get_forecast, haversine_km, search_attractions, search_restaurants

load_dotenv()
MAIN = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAST = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
FX = {"INR": 1.0, "USD": 0.012, "EUR": 0.011, "GBP": 0.0095}  # rough, for simulated prices
MAX_REVISIONS = 3


class State(TypedDict, total=False):
    request: dict
    plan: dict
    prefs: dict
    flight: dict
    hotel: dict
    weather: dict
    activities: dict
    budget: dict
    itinerary: dict
    critique: dict
    iteration: int


# ---------- helpers ----------
def ask_json(system: str, user: str, fast: bool = False) -> dict:
    llm = ChatGroq(model=FAST if fast else MAIN, temperature=0.2,
                   model_kwargs={"response_format": {"type": "json_object"}})
    out = llm.invoke([("system", system + "\nReply with one valid JSON object only."), ("user", user)])
    return json.loads(out.content)


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def norm(vals):
    lo, hi = min(vals), max(vals)
    return [0.5 if hi == lo else (v - lo) / (hi - lo) for v in vals]


def rank(items, key_weights):
    """Algorithm C - weighted ranking. key_weights: {field: weight}; negative weight = lower is better."""
    cols = {k: norm([i[k] for i in items]) for k in key_weights}
    for idx, it in enumerate(items):
        it["score"] = round(sum((w if w > 0 else -w) * (cols[k][idx] if w > 0 else 1 - cols[k][idx])
                                for k, w in key_weights.items()), 3)
    return sorted(items, key=lambda x: -x["score"])


def nearest_neighbour(start, spots):
    """Algorithm D - greedy nearest-neighbour route."""
    left, route, cur = list(spots), [], start
    while left:
        nxt = min(left, key=lambda s: haversine_km(cur, {"lat": s["lat"], "lon": s["lon"]}))
        route.append(nxt)
        left.remove(nxt)
        cur = {"lat": nxt["lat"], "lon": nxt["lon"]}
    return route


# ---------- agents ----------
def orchestrator_agent(s: State):
    """Algorithm A - decompose the request and route tasks to specialist agents."""
    r = s["request"]
    try:
        plan = ask_json(
            "You are the orchestrator of a travel-planning multi-agent system. Split the request into tasks "
            "and route each to one agent from: preference, flight, hotel, activities, weather, budget, scheduler, critic. "
            'Return {"tasks":[{"agent":"...","goal":"one short sentence"}]}.',
            json.dumps(r), fast=True)
    except Exception:
        plan = {"tasks": []}
    return {"plan": plan, "iteration": 0}


def preference_agent(s: State):
    r = s["request"]
    try:
        inf = ask_json(
            "Read the traveller's free-text wishes. Return {\"interests\":[from: food,culture,history,nature,adventure,"
            "shopping,nightlife,relaxation],\"pace\":\"relaxed|balanced|packed\",\"diet\":\"string or none\","
            "\"constraints\":\"string or none\",\"summary\":\"one sentence\"}.",
            r.get("wishes") or "no preferences given", fast=True)
    except Exception:
        inf = {}
    start, end = date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"])
    days = (end - start).days + 1
    if days < 1 or days > 14:
        raise ValueError("Trip length must be between 1 and 14 days.")
    dest = geocode(r["destination"])
    if not dest:
        raise ValueError(f"Could not find destination '{r['destination']}'.")
    origin = geocode(r["origin"]) if r.get("origin") else None
    if r.get("origin") and not origin:
        raise ValueError(f"Could not find origin '{r['origin']}'.")
    prefs = {**r, "days": days, "dest_geo": dest, "origin_geo": origin,
             "fx": FX.get(r.get("currency", "INR"), 1.0),
             "interests": inf.get("interests") or ["culture", "food"],
             "pace": inf.get("pace") if inf.get("pace") in ("relaxed", "balanced", "packed") else "balanced",
             "diet": inf.get("diet", "none"), "constraints": inf.get("constraints", "none"),
             "summary": inf.get("summary", "")}
    return {"prefs": prefs}


def flight_agent(s: State):
    p = s["prefs"]
    if not p["origin_geo"]:
        return {"flight": {"skipped": True, "options": [], "note": "No origin given - transport skipped."}}
    km = haversine_km(p["origin_geo"], p["dest_geo"])
    fx = p["fx"]
    if km < 400:  # short trip: ground transport
        base = [("Train (AC)", 0, km / 60 * 60 + 20, 0.9), ("Bus (sleeper)", 0, km / 45 * 60, 0.6), ("Cab", 0, km / 55 * 60, 2.4)]
        rate, mode = 3.0, "ground"
    else:
        base = [("Budget carrier", 1, km / 800 * 60 + 40 + 120, 0.8), ("Standard carrier", 0, km / 800 * 60 + 40, 1.0),
                ("Premium carrier", 0, km / 800 * 60 + 40, 1.7)]
        rate, mode = 4.2, "air"
    rnd = random.Random(p["origin"] + p["destination"])
    opts = [{"name": n, "stops": st, "duration_min": round(d), "mode": mode,
             "price_pp": round((1500 + km * rate) * m * 2 * fx * rnd.uniform(0.95, 1.1))}  # round trip, per person
            for n, st, d, m in base]
    ranked = rank(opts, {"price_pp": -0.5, "duration_min": -0.3, "stops": -0.2})
    return {"flight": {"options": ranked, "distance_km": round(km), "simulated": True}}


def hotel_agent(s: State):
    p = s["prefs"]
    city, fx = p["dest_geo"]["name"], p["fx"]
    rnd = random.Random(city)
    tiers = [("Backpackers Hostel", 1200), ("Comfort Inn", 2800), ("Grand Residency", 5500), ("Palace Resort", 12000)]
    opts = [{"name": f"{city} {n}", "nightly": round(b * fx * rnd.uniform(0.9, 1.15)),
             "rating": round(rnd.uniform(3.6, 4.7), 1), "distance_km": round(rnd.uniform(0.5, 7), 1)} for n, b in tiers]
    return {"hotel": {"options": rank(opts, {"nightly": -0.4, "rating": 0.4, "distance_km": -0.2}), "simulated": True}}


def activities_agent(s: State):
    """ReAct agent: reasons, calls search tools, observes, and repeats until it has a shortlist."""
    p = s["prefs"]
    g = p["dest_geo"]
    msg = (f"Destination {g['name']} at lat={g['lat']}, lon={g['lon']}. Interests: {p['interests']}. Diet: {p['diet']}. "
           f"Call the tools, then pick up to {max(p['days'] * 4, 8)} attractions that match the interests and up to 8 restaurants. "
           'Copy lat/lon exactly from tool results. Final answer: JSON only: {"attractions":[{"name":"","lat":0,"lon":0,'
           '"why":"short","indoor":true}],"restaurants":[{"name":"","cuisine":""}]}')
    try:
        agent = create_react_agent(ChatGroq(model=MAIN, temperature=0.2), [search_attractions, search_restaurants])
        res = agent.invoke({"messages": [("system", "You are a local-experiences research agent. Use tools; never invent places."),
                                         ("user", msg)]})
        data = extract_json(res["messages"][-1].content)
        data["attractions"] = [a for a in data.get("attractions", []) if "lat" in a and "lon" in a]
        if not data["attractions"]:
            raise ValueError("empty")
    except Exception:  # fallback: call tools directly
        at = search_attractions.invoke({"lat": g["lat"], "lon": g["lon"]})
        rs = search_restaurants.invoke({"lat": g["lat"], "lon": g["lon"]})
        data = {"attractions": [{**a, "why": "Nearby landmark", "indoor": False} for a in at[:p["days"] * 3]],
                "restaurants": rs[:8], "fallback": True}
    return {"activities": data}


def weather_agent(s: State):
    p = s["prefs"]
    f = get_forecast(p["dest_geo"]["lat"], p["dest_geo"]["lon"], p["start_date"], p["end_date"])
    rainy = [d["date"] for d in f["days"] if (d["rain_mm"] or 0) >= 5]
    try:
        out = ask_json('Summarise this forecast for a traveller. Return {"summary":"2 sentences","packing":["3-5 items"]}.',
                       json.dumps(f["days"]), fast=True)
    except Exception:
        out = {"summary": "Forecast loaded.", "packing": []}
    return {"weather": {**f, "rainy_dates": rainy, **out}}


def budget_agent(s: State):
    p, n = s["prefs"], s["prefs"]["travelers"]
    nights, rooms, fx = max(p["days"] - 1, 1), math.ceil(n / 2), p["fx"]
    flights = s["flight"]["options"] or [{"name": "None", "price_pp": 0}]
    hotels = s["hotel"]["options"]
    other = round((1200 + 400 + 600) * fx * n * p["days"])  # food + local transport + activities estimate
    combos = sorted(((fi + hi, f, h) for fi, f in enumerate(flights) for hi, h in enumerate(hotels)), key=lambda c: c[0])
    pick = None
    for _, f, h in combos:  # best-ranked combination that fits the budget
        if f["price_pp"] * n + h["nightly"] * nights * rooms + other <= p["budget"]:
            pick = (f, h)
            break
    over = pick is None
    if over:
        pick = (min(flights, key=lambda x: x["price_pp"]), min(hotels, key=lambda x: x["nightly"]))
    f, h = pick
    bd = {"Transport": f["price_pp"] * n, "Stay": h["nightly"] * nights * rooms,
          "Food, local transport & activities": other}
    total = sum(bd.values())
    return {"budget": {"flight": f, "hotel": h, "breakdown": bd, "total": total, "limit": p["budget"],
                       "remaining": p["budget"] - total, "over_budget": over, "currency": p["currency"]}}


def scheduler_agent(s: State):
    p, a, w, b = s["prefs"], s["activities"], s["weather"], s["budget"]
    per_day = {"relaxed": 2, "balanced": 3, "packed": 4}[p["pace"]]
    spots = a["attractions"][: per_day * p["days"]]
    route = nearest_neighbour(p["dest_geo"], spots)  # Algorithm D
    groups = [route[i * per_day:(i + 1) * per_day] for i in range(p["days"])]
    start = date.fromisoformat(p["start_date"])
    wx = {d["date"]: d for d in w["days"]}
    days_in = [{"day": i + 1, "date": date.fromordinal(start.toordinal() + i).isoformat(),
                "stops": [x["name"] for x in g], "weather": wx.get(date.fromordinal(start.toordinal() + i).isoformat())}
               for i, g in enumerate(groups)]
    feedback = (s.get("critique") or {}).get("issues", [])
    it = ask_json(
        "You are an itinerary scheduler. Build a realistic day-by-day plan. Use ONLY the given stops (in the given order "
        "within each day; you may swap whole days to put indoor places on rainy days) and the given restaurants for meals. "
        f"Day 1 begins with arrival, last day ends with departure. Respect diet='{p['diet']}', pace='{p['pace']}'. "
        f"Estimate costs in {p['currency']}. Fix these reviewer issues if any: {feedback}. "
        'Return {"days":[{"day":1,"date":"YYYY-MM-DD","theme":"","items":[{"time":"HH:MM","title":"","type":"attraction|meal|transit|rest",'
        '"notes":"short","est_cost":0}]}],"tips":["2-4 short tips"]}.',
        json.dumps({"days": days_in, "restaurants": a.get("restaurants", []), "hotel": b["hotel"]["name"],
                    "transport": b["flight"]["name"], "indoor": [x["name"] for x in spots if x.get("indoor")]}))
    return {"itinerary": it, "iteration": s.get("iteration", 0) + 1}


def critic_agent(s: State):
    p, it, b = s["prefs"], s["itinerary"], s["budget"]
    issues, warnings = [], []
    days = it.get("days", [])
    if len(days) != p["days"]:
        issues.append(f"Plan has {len(days)} days but the trip is {p['days']} days.")
    for d in days:
        items = d.get("items", [])
        times = [i.get("time", "") for i in items]
        if len(items) < 2:
            issues.append(f"Day {d.get('day')} has too few items.")
        if times != sorted(times):
            issues.append(f"Day {d.get('day')} times are out of order.")
    try:
        rev = ask_json('Review this travel plan for REAL problems only: outdoor activity on heavy-rain days, impossible timing, '
                       f"diet violations (diet={p['diet']}), too packed for pace={p['pace']}. "
                       'Return {"issues":["short issue"]} (empty list if fine).',
                       json.dumps({"plan": days, "rainy_dates": s["weather"]["rainy_dates"]}), fast=True)
        issues += rev.get("issues", [])[:3]
    except Exception:
        pass
    if b["over_budget"]:
        warnings.append("Even the cheapest options exceed the budget. Consider fewer days or a higher budget.")
    return {"critique": {"approved": not issues, "issues": issues, "warnings": warnings}}


def after_critic(s: State):
    return "scheduler_agent" if (not s["critique"]["approved"] and s.get("iteration", 0) < MAX_REVISIONS) else END


# ---------- graph ----------
def build():
    g = StateGraph(State)
    for name, fn in [("orchestrator_agent", orchestrator_agent), ("preference_agent", preference_agent),
                     ("flight_agent", flight_agent), ("hotel_agent", hotel_agent),
                     ("activities_agent", activities_agent), ("weather_agent", weather_agent),
                     ("budget_agent", budget_agent), ("scheduler_agent", scheduler_agent), ("critic_agent", critic_agent)]:
        g.add_node(name, fn)
    g.add_edge(START, "orchestrator_agent")
    g.add_edge("orchestrator_agent", "preference_agent")
    par = ["flight_agent", "hotel_agent", "activities_agent", "weather_agent"]
    for n in par:
        g.add_edge("preference_agent", n)
    g.add_edge(par, "budget_agent")
    g.add_edge("budget_agent", "scheduler_agent")
    g.add_edge("scheduler_agent", "critic_agent")
    g.add_conditional_edges("critic_agent", after_critic, {"scheduler_agent": "scheduler_agent", END: END})
    return g.compile()


graph = build()
