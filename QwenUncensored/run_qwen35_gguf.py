import time
import csv
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama


CSV_PATH = Path(__file__).with_name("qwen_runs.csv")

# Configure these before running the script.
PROMPT = "Replace this string with your prompt."
MODEL_PATH = Path(r"path\to\your\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
CTX = 131072
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
N_PREDICT = 512
N_GPU_LAYERS = None  # e.g. 99, or None to let llama.cpp decide.

_LLM: Llama | None = None


def _get_llm() -> Llama:
    global _LLM
    if _LLM is not None:
        return _LLM

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    _LLM = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CTX,
        n_gpu_layers=N_GPU_LAYERS or 0,
        logits_all=False,
        embedding=False,
    )
    return _LLM


def run_qwen(prompt: str | None = None) -> tuple[str, float]:
    """Run the Qwen model for the given prompt and return (output_text, latency_seconds)."""
    if prompt is None:
        prompt = PROMPT.strip()
    if not prompt:
        raise ValueError("PROMPT is empty. Set PROMPT at the top of run_qwen35_gguf.py or pass a prompt.")

    llm = _get_llm()

    start = time.time()
    result = llm(
        prompt,
        max_tokens=N_PREDICT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )
    latency = time.time() - start

    # llama-cpp-python returns either "choices"[0]["text"] or "choices"[0]["message"]["content"]
    choice = result["choices"][0]
    output_text = choice.get("text") or choice.get("message", {}).get("content", "")
    output_text = (output_text or "").strip()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "prompt", "output_text", "latency_seconds"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                prompt,
                output_text,
                f"{latency:.3f}",
            ]
        )

    return output_text, latency


def main() -> int:
    try:
        output_text, latency = run_qwen()
        print(output_text)
        print(f"\n[latency] {latency:.3f} seconds", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
