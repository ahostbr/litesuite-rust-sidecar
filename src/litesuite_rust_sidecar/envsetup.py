"""Builds one engine environment by its recipe (C3), from inside that environment.

The app has already made the environment (`uv venv --python 3.11 <env>`) and installed this
package into it (`uv pip install litesuite-rust-sidecar[litesound]==<version>`). This runs the
recipe's ordered `uv pip` steps, then the verify program, and writes `litesound-env.json` in the
environment only if verify passed, so a failed or cancelled build counts as not installed.

    python -m litesuite_rust_sidecar.envsetup <env> --uv <path to uv.exe> [--app litesound] [--dry-run]

`--dry-run` prints the exact commands in order and runs none of them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MARKER = "litesound-env.json"

# The GPU check of LiteSound's setup_speech_envs.py. torch.cuda.is_available() is not the test: it
# is True on a build with no kernels for this card. The device's own compute capability must be in
# torch.cuda.get_arch_list() and a real kernel must run. It touches the GPU for a moment and loads
# no model.
GPU_PROBE = r"""
import torch, sys
if not torch.cuda.is_available():
    print("WARN no CUDA device visible; backends will run on CPU")
    sys.exit(0)
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
name = torch.cuda.get_device_name(0)
if cap not in arches:
    print("FAIL %s is %s but this torch (%s) builds only %s" % (name, cap, torch.__version__, arches))
    sys.exit(1)
x = torch.randn(512, 512, device="cuda")
float((x @ x).sum())
print("%s %s | torch %s" % (name, cap, torch.__version__))
"""


def recipes(app: str = "litesound") -> dict:
    return tomllib.loads((PACKAGE / app / "envs.toml").read_text(encoding="utf-8"))["envs"]


def recipe(env: str, app: str = "litesound") -> dict:
    envs = recipes(app)
    if env not in envs:
        raise SystemExit(f"no environment {env!r} in {app}'s recipes: {', '.join(envs)}")
    found = envs[env]
    if found.get("checkout_only"):
        raise SystemExit(f"{env}: {found.get('reason', 'not installable from the package')}")
    return found


def commands(env: str, uv: str, python: str, app: str = "litesound") -> list[list[str]]:
    """The `uv pip` commands of the recipe, in order, for the interpreter `python`."""
    found = recipe(env, app)
    index = found.get("torch_index", "")
    out = []
    for step in found["steps"]:
        args = [a.replace("{torch_index}", index) for a in step]
        out.append([uv, "pip", args[0], "--python", python, *args[1:]])
    return out


def verify(env: str, python: str, app: str = "litesound") -> tuple[bool, str]:
    """The GPU check, then the recipe's import check. Returns (passed, the last line said)."""
    found = recipe(env, app)
    gpu = subprocess.run([python, "-c", GPU_PROBE], capture_output=True, text=True)
    said = (gpu.stdout or gpu.stderr).strip().splitlines()
    detail = said[-1] if said else "no output"
    if gpu.returncode != 0:
        return False, detail
    program = found.get("verify", "")
    if program:
        program = program.replace("{vendor}", str(PACKAGE / app / "vendor"))
        imported = subprocess.run([python, "-c", program], capture_output=True, text=True)
        if imported.returncode != 0:
            last = (imported.stderr or "").strip().splitlines()
            return False, f"import failed: {last[-1] if last else 'unknown'}"
    return True, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m litesuite_rust_sidecar.envsetup")
    parser.add_argument("env")
    parser.add_argument("--uv", required=True, help="the uv.exe the app ships")
    parser.add_argument("--app", default="litesound")
    parser.add_argument("--dry-run", action="store_true", help="print the commands in order; run nothing")
    args = parser.parse_args(argv)
    python = sys.executable
    steps = commands(args.env, args.uv, python, args.app)
    if args.dry_run:
        for step in steps:
            print(subprocess.list2cmdline(step))
        return 0
    marker = Path(sys.prefix) / MARKER
    marker.unlink(missing_ok=True)
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {subprocess.list2cmdline(step)}", flush=True)
        if subprocess.run(step).returncode != 0:
            print(f"FAILED at step {i}", flush=True)
            return 1
    passed, detail = verify(args.env, python, args.app)
    print(("verified: " if passed else "verify FAILED: ") + detail, flush=True)
    if not passed:
        return 1
    try:
        version = metadata.version("litesuite-rust-sidecar")
    except metadata.PackageNotFoundError:
        version = "unknown"
    marker.write_text(json.dumps({"app": args.app, "env": args.env, "package_version": version, "torch_index": recipe(args.env, args.app).get("torch_index", ""), "verified": detail}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
