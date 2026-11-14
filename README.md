# litesuite-rust-sidecar

The Python engines that the Rust Lite apps start beside themselves, as one package with one
sub-package per app. Today it holds `litesuite_rust_sidecar.litesound`, LiteSound's audio engine,
moved from LiteSound's own `python/audio_engine` folder with only its imports changed.

Not published to PyPI yet.

## What the move changed

Imports go from script-relative (`from backends.musicgen import ...`) to package-relative
(`from .backends.musicgen import ...`), and the `if __name__ == "__main__"` block of `server.py`
is now `__main__.py`. The nine HTTP routes, their bodies, the environment variables and every
backend are unchanged.

Start the engine: `python -m litesuite_rust_sidecar.litesound` (uvicorn on 127.0.0.1, port from
`LITESOUND_PORT`).

## Tests

```
python -m unittest discover -s tests
```

They load no model and need no torch.

## Known

- When `LITESOUND_MODELS_DIR`, `LITESOUND_LOG_DIR` and the runtime folder variables are not set,
  the engine falls back to folders counted upwards from its own file. The Rust app always sets
  these variables.
- The launch token (`auth.py`): with `LITESUITE_SIDECAR_TOKEN` set, every request but `/health` needs
  the header `X-LiteSuite-Sidecar-Token`; unset, the engine is open as before.
- Environments: `litesound/envs.toml` holds the ordered `uv pip` steps of each engine environment,
  and `python -m litesuite_rust_sidecar.envsetup <env> --uv <uv.exe> [--dry-run]` runs them, checks
  the GPU and the imports, and writes `litesound-env.json` only when that passes. `yue` is not
  installable from the package yet (no recorded recipe).
- MIT (`LICENSE`). The vendored VibeVoice tree keeps its own licence and `PROVENANCE.md`.
