# VibeVoice loader — provenance, review, and dependency resolution

Status: **VENDORED, reviewed, and running.** This file is the deliverable of WS1 T1/T2.

## Source

| | |
|---|---|
| repo | `https://github.com/microsoft/VibeVoice` |
| pinned commit | `94da20d98b2fa7688e9cbfaf7692ddb4954f7600` (2026-07-24T09:24:29Z) |
| retrieved | 2026-08-07 |
| method | `https://github.com/microsoft/VibeVoice/archive/<sha>.tar.gz`, sha256 `6f54cf9e3620a608092a86237e875fe8060fea54555608bca4bc3875984d0c76` |
| vendored | the `vibevoice/` package only — 22 `.py` files + 2 config JSON + LICENSE, 345 KB |
| license | MIT (`LICENSE`, vendored alongside) |
| interpreter | `python/.venv-vibevoice` — transformers **4.51.3**, torch cu128 nightly |

### The "Microsoft pulled it" claim is false in BOTH halves

A widely-shared video (Helena Liu, `31bZOIk53_I`) says Microsoft removed VibeVoice and
that you must clone a community mirror. Measured 2026-08-07:

- `GET api.github.com/repos/microsoft/VibeVoice` → **200 OK**, `archived:false`,
  `disabled:false`, 52,171 stars, last push 2026-07-24. The **code** repo is live.
- The **weights** are live and ungated too.

⚠️ **Do NOT `pip install vibevoice`.** PyPI `vibevoice` 0.0.1 (uploaded 2025-09-26) is an
unaffiliated third-party backup that self-describes as *"a backup of the code"* — while
carrying a Microsoft address as its author and Microsoft's own GitHub URLs in
`project_urls`. It looks first-party at a glance. It is not, and its single-version
history is the classic supply-chain shape. Vendoring from the pinned commit above avoids
it entirely.

🔎 Curiosity worth knowing: `modular/modeling_vibevoice.py` line 1 reads
`# copied from https://github.com/vibevoice-community/VibeVoice/...`. Microsoft's own repo
credits the *community* fork for that file — the two have cross-pollinated. Vendoring from
the first-party pinned commit is still the right call, but the lineage is not one-way.

## 🔴 CHECKPOINT: Realtime-0.5B, not 1.5B

This is the single most important fact here, and it contradicts the original plan.

At commit `94da20d9` the **only inference-capable class shipped is
`VibeVoiceStreamingForConditionalGenerationInference`**. `modeling_vibevoice.py` still
defines `VibeVoiceForConditionalGeneration`, but it is the base/training class — it does
not inherit `GenerationMixin` and has no `generate()` producing `speech_outputs`.
Microsoft removed `modeling_vibevoice_inference.py` from the repo.

| model | config `architectures` | usable at this pin? |
|---|---|---|
| `microsoft/VibeVoice-1.5B` (5.04 GB) | `VibeVoiceForConditionalGeneration` | ❌ inference subclass no longer exists |
| **`microsoft/VibeVoice-Realtime-0.5B` (1.90 GB)** | **`VibeVoiceStreamingForConditionalGenerationInference`** | ✅ exactly the vendored class |

The registry therefore points at **Realtime-0.5B**. Pointing it at 1.5B would download
5 GB that the vendored loader cannot load. The class is named "streaming", but the backend
passes no streamer and consumes the complete result — batch semantics, file-on-disk
contract, no streaming transport, exactly as planned.

## Security review (performed before the files were copied)

22 `.py` files, 345 KB — small enough to audit properly, which is the whole argument for
vendoring over cloning a 120 MB mirror.

Scanned for `subprocess`, `os.system`, `eval`, `exec`, `__import__`, `pickle.load`,
`marshal`, `base64.b64decode`, sockets, `requests`/`urllib`/`httpx`, and
telemetry SDKs (sentry / posthog / wandb / mlflow). Findings:

| finding | verdict |
|---|---|
| 5 hits on `\beval\s*\(` | **false positives** — all `nn.Module.eval()`, model mode |
| `processor/audio_utils.py`: `from subprocess import run` | **benign** — invokes `ffprobe`/`ffmpeg` with a **fixed argv list**, no `shell=True`, no string interpolation, `-nostdin` set. No injection surface. |
| `vibevoice_tokenizer_processor.py:225`: `torch.load(..., weights_only=True)` | **already the safe form** — blocks arbitrary pickle execution |
| ~25 hardcoded URLs | **all citations** — arxiv, HF papers, GitHub issues, in comments/docstrings. Zero runtime network calls. |
| telemetry | **none** |
| `vibevoice/__init__.py` | re-exports only; no import-time side effects |

### Real dependency set (read from the code, not from PyPI metadata)

PyPI's metadata lists gradio, aiortc, av, ml-collections, absl-py, numba… The vendored
code actually imports only:

```
torch  transformers  diffusers  librosa  numpy  soundfile  tqdm  apex
```

`apex` is optional — wrapped in `try/except ImportError` with a native fallback, and
disabled by default anyway (`OPTIMIZE_FOR_SPEED` defaults to `0`). Skipped. So the install
is far smaller than the package's declared deps imply, and gradio/aiortc/av never land.

### SHA256 manifest

| file | sha256 |
|---|---|
| `__init__.py` | `d2545b8b4f70348514a367a04ae38d0da3de98174a88745cdb5e64bb0e4b6839` |
| `LICENSE` | `98a96c38494df4963951e32b81ec4effebe4c8a812ab8d9e7d185d59e86fe2fc` |
| `configs/qwen2.5_1.5b_64k.json` | `6d65b90d0703453ce9aa27de2cbca87e0cc276f82acfc0d388538400e429238c` |
| `configs/qwen2.5_7b_32k.json` | `a890a296f0c62bc8d62ead244fdd6a3738d2a67dc6c0634fe9aa3f8064047c5e` |
| `modular/__init__.py` | `a5453c864eedead47a9d52a03d78e8893bb9983dc3e93ca9ef1205e6803d7448` |
| `modular/configuration_vibevoice.py` | `bcf59d8749405d5d25436923444436f6719cb9b2b403941a8a5e46f5b94873d0` |
| `modular/configuration_vibevoice_streaming.py` | `1b4417ea2ffe5b42a58d1ee6b3c9508bc253930c2d55f8ec1f10916e36b8029e` |
| `modular/modeling_vibevoice.py` | `6a7fbe0f967f09cd05c2cbbd270276bac8e90a6061510a82029cbf8bc680caf2` |
| `modular/modeling_vibevoice_asr.py` | `b460fe1dad02d43203470c8b9a79492dae540ba2d91b982ee9639f880e114919` |
| `modular/modeling_vibevoice_streaming.py` | `96d402716fe2cc6284c4ad7f8b2309dd8d06d2133681e266387466116e908afb` |
| `modular/modeling_vibevoice_streaming_inference.py` | `cb5360a62dbf3a836cea763348bc6330c4d7a422fbe572f4abf7b3c7a15bce8b` |
| `modular/modular_vibevoice_diffusion_head.py` | `8a90ac685b9e32793749e2c58b48906758486d8b0bc0d92c8dfd141aa053852e` |
| `modular/modular_vibevoice_text_tokenizer.py` | `9f188a3b1d21fbeb8eb26b62693a8ffa1024edff29af28c5d8599c9ade143f7f` |
| `modular/modular_vibevoice_tokenizer.py` | `f2fdcdaff26a685510c982634cad3be8258b299f89566a87188c932fd85f816d` |
| `modular/streamer.py` | `e6129f46e2c61466ca96ce479c04adfaad32f483230f4d3531c96717bd16f603` |
| `processor/__init__.py` | `957a1c1b18310691647cc0adaece1d8f940785deaa4d8d17ece2059bee0b9dc8` |
| `processor/audio_utils.py` | `e15f184c18f754862e38082836b9c19c0f97c03c939412ab856838dc30788bcf` |
| `processor/vibevoice_asr_processor.py` | `e5551319fac744b5152202a8e41888dad2d94d3fa941aa755d0888caa5a6a4ab` |
| `processor/vibevoice_processor.py` | `5b28fa5247a83cb65385e06428fb112369f50723e343d7b38c7a6e7e1dfddab5` |
| `processor/vibevoice_streaming_processor.py` | `41ff57a7da668db9099c395ab7bbee9b2cd7aa0c2f8f0b0c214768c051aa5a53` |
| `processor/vibevoice_tokenizer_processor.py` | `3b18571d3b5cac91211412f03c3a8477bb0242173a4a53a633498cac470f1068` |
| `schedule/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `schedule/dpm_solver.py` | `be10eae978afda40a6bc082806fa1e8f0f4b4e451fd1d21cec97347116ffba73` |
| `schedule/timestep_sampler.py` | `70b003bdfdc716d09942ecfcd0e2f438bac27ea16da9527e105e008a0069dd83` |
| `scripts/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Voices are `.pt` prefills, not wavs

Unlike Chatterbox, VibeVoice does not clone from an arbitrary audio clip. It conditions on
a **pre-computed prefill tensor** fetched by the repo's `demo/download_experimental_voices.sh`.
11 English voices (en1 + en2 bundles) are installed to `bin/models/vibevoice-voices/`;
override the location with `LITESOUND_VIBEVOICE_VOICES_DIR`. Handing the backend a `.wav`
is a usage error and is reported as one.

### 🔒 Why the backend audits those pickles

`torch.load(..., weights_only=True)` **cannot read these files** — not because they are
malicious, but because transformers' `BaseModelOutputWithPast` subclasses `OrderedDict`
and torch's safe unpickler only permits `SETITEMS` on *exact* `dict`/`OrderedDict`/`Counter`.
The first-party demo works around it with `safe_globals`, which does not help for this
particular failure.

Blindly flipping to `weights_only=False` would mean arbitrary code execution on a
downloaded file. Instead `backends/vibevoice.py::audit_prefill_pickle()` moves the
protection earlier: it walks the pickle's opcodes with `pickletools`, enumerates every
`GLOBAL` it imports, refuses outright on `STACK_GLOBAL` (unauditable statically), and
loads only if every symbol is allowlisted.

Audited across all 11 prefills — exactly five symbols, all inert data containers:

```
collections.OrderedDict · torch.BFloat16Storage · torch._utils._rebuild_tensor_v2
transformers.cache_utils.DynamicCache · transformers.modeling_outputs.BaseModelOutputWithPast
```

No `os`, no `subprocess`, no `builtins.eval`, no `__reduce__` gadget. Verified in both
directions: 11/11 real prefills pass, and a purpose-built `os.system` pickle is **refused**.

## Dependency resolution — why the isolation exists at all

Measured with `pip install --dry-run` (resolves without mutating) against the main
`python/.venv` (Python 3.11.9, torch 2.12.0.dev+cu128, transformers 5.6.1):

| package | verdict |
|---|---|
| `chatterbox-tts` 0.1.7 | 🔴 **SEVERE.** Hard-pins `torch==2.6.0` / `torchaudio==2.6.0` / `transformers==5.2.0` / `diffusers==0.29.0` / `numpy<2` — 68 packages. torch 2.6.0 is CUDA 12.4 with **no `sm_120` kernels**, so it would take the **RTX 5090 off CUDA for every other backend** (MusicGen, Stable Audio, Bark, AudioLDM2, ACE-Step) at once. |
| `vibevoice` (PyPI) | Moderate — torch survives, but transformers 5.6.1 → **4.51.3**, tokenizers → 0.21.4, accelerate → 1.6.0. |

**The plan had this backwards.** It treated Chatterbox as the safe default whose
"dependency story is confirmed". It was confirmed in *LiteSuite's* venv, a different
environment with different pins — the confirmation never transferred to LiteSound.

⭐ **`pip install --dry-run` exited 0 for both.** Exit 0 means "a resolution was found",
not "safe" — pip found one *by downgrading torch*. **The verdict is in the `Would install`
list, not the exit code.**

## The install recipe (order matters)

Isolation alone is insufficient: torch 2.6.0 in an isolated venv still cannot drive a
5090. Install torch FIRST, then the package with `--no-deps`, so its pin never applies.

```
# Chatterbox
python -m venv python/.venv-speech
.venv-speech/Scripts/python -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
.venv-speech/Scripts/python -m pip install --no-deps chatterbox-tts
.venv-speech/Scripts/python -m pip install "numpy<2" "librosa==0.11.0" s3tokenizer "transformers==5.2.0" \
    "diffusers==0.29.0" "resemble-perth>=1.0.0" "conformer==0.3.2" "safetensors==0.5.3" spacy-pkuseg \
    "pykakasi==2.3.0" pyloudnorm omegaconf scipy          # gradio skipped — demo UI only

# VibeVoice (loader is vendored; the PyPI package is never installed)
python -m venv python/.venv-vibevoice
.venv-vibevoice/Scripts/python -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
.venv-vibevoice/Scripts/python -m pip install "transformers==4.51.3" diffusers librosa soundfile tqdm accelerate
```

pip then *warns* `requires torch==2.6.0 ... incompatible`. **That warning is the intended
override, not a failure.** The proof of success is not `torch.cuda.is_available()` (true
even on a CPU fallback) but `torch.cuda.get_arch_list()` containing **`sm_120`** plus a
real kernel executing on device. Both venvs verified that way.

## Measured results (RTX 5090)

| | Chatterbox | VibeVoice Realtime-0.5B |
|---|---|---|
| venv | `.venv-speech` (4.93 GB) | `.venv-vibevoice` |
| weights | ~3 GB of `ResembleAI/chatterbox` | 1.90 GB |
| load | 11.6 s | 9.2 s |
| speed | faster than realtime | RTF **0.44–0.82x** |
| output | 24 kHz mono **float32** (`subtype=FLOAT`) | 24 kHz mono **PCM_16** |
| conditioning | any reference **wav**, zero-shot | pre-computed **`.pt`** prefill |

⚠️ Chatterbox's float32 output cannot be opened by Python's stdlib `wave`
(`unknown format: 3`); use `soundfile`. VibeVoice writes PCM_16 and is the more
interchange-friendly of the two for game audio. Worth considering whether Chatterbox
should write PCM_16 too — it would diverge from `bark.py`/`musicgen.py`, so it is
deliberately left alone pending a decision.

ℹ️ VibeVoice clip lengths quantize to 3200-sample (0.1333 s) frames — two similar-length
lines landing on an identical duration is that quantization, not truncation or padding.
