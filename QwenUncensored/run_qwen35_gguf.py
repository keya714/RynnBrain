import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


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


def _read_prompt_from_stdin() -> str:
    data = sys.stdin.read()
    return data.strip("\r\n")


def _build_cmd(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [_resolve_exe(args.llama_cli)]

    cmd += ["-m", args.model]
    cmd += ["--jinja"]

    # Context and generation knobs (match HF card recommendations reasonably).
    cmd += ["-c", str(args.ctx)]
    cmd += ["--temp", str(args.temperature)]
    cmd += ["--top-p", str(args.top_p)]
    cmd += ["--top-k", str(args.top_k)]

    if args.n_predict is not None:
        cmd += ["-n", str(args.n_predict)]

    if args.n_gpu_layers is not None:
        cmd += ["-ngl", str(args.n_gpu_layers)]

    if args.mmproj:
        cmd += ["--mmproj", args.mmproj]

    # Chat-ish formatting: this is the simplest cross-template approach for llama.cpp.
    cmd += ["-p", args.prompt]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local GGUF of HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive via llama.cpp (llama-cli)."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the downloaded GGUF model file (e.g. ...Q4_K_M.gguf).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, prompt is read from stdin.",
    )
    parser.add_argument(
        "--llama-cli",
        default="llama-cli",
        help="Path to llama-cli (or llama-cli.exe) from llama.cpp, or name on PATH.",
    )
    parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional mmproj GGUF for vision models (only if using image/video prompts).",
    )
    parser.add_argument("--ctx", type=int, default=131072, help="Context length (-c).")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument(
        "--n-predict",
        type=int,
        default=512,
        help="Max tokens to generate (-n).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="GPU layers to offload (-ngl). Example: 99. Omit to let llama.cpp default.",
    )

    args = parser.parse_args()

    if not args.prompt:
        args.prompt = _read_prompt_from_stdin()

    if not args.prompt:
        print("Error: empty prompt. Provide --prompt or pipe text to stdin.", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return 2

    if args.mmproj and not Path(args.mmproj).exists():
        print(f"Error: mmproj file not found: {args.mmproj}", file=sys.stderr)
        return 2

    cmd = _build_cmd(args)

    # Print the command so it's easy to reproduce/debug.
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run] {printable}", file=sys.stderr)

    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except FileNotFoundError:
        print(
            "Error: llama-cli not found. Build/install llama.cpp and pass --llama-cli path to llama-cli(.exe).",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
