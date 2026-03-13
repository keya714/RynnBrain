from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_qwen35_gguf import run_qwen


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

