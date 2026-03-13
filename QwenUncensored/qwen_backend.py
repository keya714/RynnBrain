from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_qwen35_gguf import run_qwen


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://8amc6jr91c2x3j-8000.proxy.runpod.net",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate")
async def generate(body: PromptRequest):
    output_text, latency = run_qwen(body.prompt)
    return {
        "output": output_text,
        "latency_seconds": latency,
    }

