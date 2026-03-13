from fastapi import FastAPI
from pydantic import BaseModel

from run_qwen35_gguf import run_qwen


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI()


@app.post("/api/generate")
async def generate(body: PromptRequest):
    output_text, latency = run_qwen(body.prompt)
    return {
        "output": output_text,
        "latency_seconds": latency,
    }

