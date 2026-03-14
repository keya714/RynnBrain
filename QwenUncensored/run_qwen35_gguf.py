import os
import shlex
import subprocess
import sys
import time
import csv
from datetime import datetime
from pathlib import Path


CSV_PATH = Path(__file__).with_name("qwen_runs.csv")

# Configure these before running the script.
PROMPT = "Replace this string with your prompt."
# Default to your Runpod path; adjust if needed.
MODEL_PATH = Path("/workspace/models/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
LLAMA_CLI = "/workspace/llama.cpp/build/bin/llama-cli"
MMPROJ_PATH = None  # Optional: Path to mmproj GGUF for vision, or None.
CTX = 131072
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
N_PREDICT = 512
N_GPU_LAYERS = None  # e.g. 99, or None to let llama.cpp decide.


def _resolve_exe(exe: str) -> str:
    # Prefer explicit path, otherwise rely on PATH.
    p = Path(exe)
    if p.exists():
        return str(p)
    if os.name == "nt" and not exe.lower().endswith(".exe"):
        p2 = Path(exe + ".exe")
        if p2.exists():
            return str(p2)
    return exe


def _build_cmd(prompt: str) -> list[str]:
    cmd: list[str] = [_resolve_exe(LLAMA_CLI)]

    cmd += ["-m", str(MODEL_PATH)]
    cmd += ["--jinja"]

    cmd += ["-c", str(CTX)]
    cmd += ["--temp", str(TEMPERATURE)]
    cmd += ["--top-p", str(TOP_P)]
    cmd += ["--top-k", str(TOP_K)]
    cmd += ["-n", str(N_PREDICT)]

    if N_GPU_LAYERS is not None:
        cmd += ["-ngl", str(N_GPU_LAYERS)]

    if MMPROJ_PATH:
        cmd += ["--mmproj", str(MMPROJ_PATH)]

    cmd += ["-p", prompt]
    return cmd


def run_qwen(prompt: str | None = None) -> tuple[str, float]:
    """Run the Qwen model for the given prompt and return (output_text, latency_seconds)."""
    if prompt is None:
        prompt = PROMPT.strip()
    if not prompt:
        raise ValueError("PROMPT is empty. Set PROMPT at the top of run_qwen35_gguf.py or pass a prompt.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if MMPROJ_PATH is not None and not Path(MMPROJ_PATH).exists():
        raise FileNotFoundError(f"mmproj file not found: {MMPROJ_PATH}")

    cmd = _build_cmd(prompt)

    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run] {printable}", file=sys.stderr)

    try:
        start = time.time()
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        latency = time.time() - start

        output_text = proc.stdout.strip()

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
    except FileNotFoundError:
        raise FileNotFoundError(
            "llama-cli not found. Build/install llama.cpp and set LLAMA_CLI to its path."
        )
    except subprocess.CalledProcessError as e:
        msg = "llama-cli exited with non-zero status."
        detail = e.stderr or e.stdout or ""
        raise RuntimeError(f"{msg}\n{detail}") from e


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
