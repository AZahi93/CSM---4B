import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="TravelGenie API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class PlanRequest(BaseModel):
    origin: str = ""
    destination: str
    start_date: str
    end_date: str
    travelers: int = 1
    budget: float = 40000
    currency: str = "INR"
    wishes: str = ""


def sse(obj):
    return f"data: {json.dumps(obj, default=str)}\n\n"


@app.post("/api/plan")
async def plan(req: PlanRequest):
    async def stream():
        try:
            async for chunk in graph.astream({"request": req.model_dump()}, stream_mode="updates"):
                for node, update in chunk.items():
                    yield sse({"type": "agent", "agent": node, "update": update or {}})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"ok": True}
