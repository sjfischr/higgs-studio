"""
Model downloader for Higgs Studio.

Downloads the Higgs Audio v3 TTS weights (transformers-ported build) and the
Moonshine ASR model used for auto-transcribing reference clips, both from
HuggingFace via huggingface_hub.

NOTE ON THE REPO CHOICE
-----------------------
The Gradio app drives the model through `model.generate_speech(...)` and
`model.get_audio_codec()`. Those are custom methods that live in the
transformers-ported repo below (loaded with trust_remote_code). Boson AI's raw
weights (bosonai/higgs-tts-3-4b) are built for vLLM-Omni / SGLang-Omni serving
and do NOT expose those methods through plain transformers, so we pull the port.
Same v3 model, repackaged to run on stock 🤗 transformers.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Generator

logger = logging.getLogger(__name__)

# --- Model repository information ---
MODELS_TO_DOWNLOAD = [
    {
        "name": "Higgs-Audio-v3-TTS-4B",
        "repo_id": "multimodalart/higgs-audio-v3-tts-4b-transformers",
        "local_dir_name": "higgs-audio-v3-tts-4b",
        "description": "Higgs Audio v3 TTS, 4B params, transformers-ported (~9 GB)",
        "size_gb": 9.0,
        "required": True,
    },
    {
        "name": "Moonshine-Base-ASR",
        "repo_id": "UsefulSensors/moonshine-base",
        "local_dir_name": "moonshine-base",
        "description": "ASR for auto-transcribing reference clips (~400 MB)",
        "size_gb": 0.4,
        "required": False,  # nice-to-have: enables auto-transcribe offline
    },
]


@dataclass
class DownloadProgress:
    """Tracks download progress for UI updates."""
    model_name: str
    status: str = "pending"  # pending, downloading, completed, failed
    progress_percent: float = 0.0
    current_file: str = ""
    error_message: Optional[str] = None

    @property
    def status_emoji(self) -> str:
        return {
            "completed": "✅",
            "downloading": "⏳",
            "failed": "❌",
        }.get(self.status, "⏸️")


@dataclass
class DownloadState:
    """Global download state for tracking across all models."""
    is_downloading: bool = False
    current_model: Optional[str] = None
    progress: dict = field(default_factory=dict)
    cancel_requested: bool = False

    def reset(self):
        self.is_downloading = False
        self.current_model = None
        self.progress = {}
        self.cancel_requested = False


_download_state = DownloadState()


def get_download_state() -> DownloadState:
    return _download_state


def check_huggingface_hub() -> tuple[bool, str]:
    """Check if huggingface_hub is installed and working."""
    try:
        import huggingface_hub
        return True, f"huggingface_hub v{huggingface_hub.__version__} installed"
    except ImportError:
        return False, "huggingface_hub not installed. Run: pip install huggingface_hub"


def _resolve_local_dir(model_dir: str, model_info: dict) -> str:
    name = model_info.get("local_dir_name") or ""
    return os.path.join(model_dir, name) if name else model_dir


def download_all_models_generator(model_dir: str) -> Generator[str, None, bool]:
    """
    Download all required models, yielding markdown status for Gradio streaming.
    Resumable: re-running picks up where an interrupted download left off.
    """
    global _download_state

    hf_ok, hf_msg = check_huggingface_hub()
    if not hf_ok:
        yield f"❌ {hf_msg}"
        return False

    from huggingface_hub import snapshot_download

    _download_state.is_downloading = True
    _download_state.cancel_requested = False
    os.makedirs(model_dir, exist_ok=True)

    total = len(MODELS_TO_DOWNLOAD)
    total_size = sum(m["size_gb"] for m in MODELS_TO_DOWNLOAD)
    all_success = True

    yield "## 🍋 Starting Model Download\n"
    yield f"**Target directory:** `{model_dir}`\n"
    yield f"**Total download size:** ~{total_size:.1f} GB\n"
    yield "---\n"

    for idx, model_info in enumerate(MODELS_TO_DOWNLOAD, 1):
        if _download_state.cancel_requested:
            yield "\n⚠️ **Download cancelled by user.**"
            _download_state.is_downloading = False
            return False

        name = model_info["name"]
        repo_id = model_info["repo_id"]
        local_dir = _resolve_local_dir(model_dir, model_info)
        _download_state.current_model = name

        yield f"\n### [{idx}/{total}] {name}\n"
        yield f"- **Description:** {model_info['description']}\n"
        yield f"- **Repository:** `{repo_id}`\n"
        yield f"- **Size:** ~{model_info['size_gb']} GB\n"
        yield f"- **Downloading to:** `{local_dir}`\n"
        yield "- **Status:** ⏳ Downloading...\n"

        try:
            os.makedirs(local_dir, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,
            )
            yield "- **Status:** ✅ **Completed!**\n"
        except Exception as e:
            err = str(e)
            logger.error(f"Failed to download {name}: {err}")
            yield f"- **Status:** ❌ **Failed:** {err}\n"
            if model_info.get("required", True):
                all_success = False
            else:
                yield "  _(optional — app still runs, auto-transcribe disabled)_\n"

    _download_state.is_downloading = False
    _download_state.current_model = None

    yield "\n---\n"
    if all_success:
        yield "## ✅ All Models Downloaded Successfully!\n"
        yield "Head to the **🎙️ Studio** tab to start synthesizing.\n"
    else:
        yield "## ⚠️ Some Required Downloads Failed\n"
        yield "Check the errors above and re-run — downloads resume where they left off.\n"

    return all_success


def download_single_model_generator(model_dir: str, model_name: str) -> Generator[str, None, bool]:
    """Download a single component by its 'name' field."""
    global _download_state

    model_info = next((m for m in MODELS_TO_DOWNLOAD if m["name"] == model_name), None)
    if not model_info:
        yield f"❌ Unknown model: {model_name}"
        return False

    hf_ok, hf_msg = check_huggingface_hub()
    if not hf_ok:
        yield f"❌ {hf_msg}"
        return False

    from huggingface_hub import snapshot_download

    _download_state.is_downloading = True
    _download_state.current_model = model_name
    repo_id = model_info["repo_id"]
    local_dir = _resolve_local_dir(model_dir, model_info)
    os.makedirs(local_dir, exist_ok=True)

    yield f"## Downloading {model_name}\n"
    yield f"- **Repository:** `{repo_id}`\n"
    yield f"- **Size:** ~{model_info['size_gb']} GB\n"
    yield f"- **Destination:** `{local_dir}`\n"
    yield "- **Status:** ⏳ Downloading...\n"

    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
        yield f"\n✅ **{model_name} downloaded successfully!**\n"
        _download_state.is_downloading = False
        return True
    except Exception as e:
        err = str(e)
        logger.error(f"Failed to download {model_name}: {err}")
        yield f"\n❌ **Failed:** {err}\n"
        _download_state.is_downloading = False
        return False


def cancel_download():
    """Request cancellation of the current download."""
    _download_state.cancel_requested = True
    logger.info("Download cancellation requested")


def get_model_info_text() -> str:
    """Formatted information about the models to be downloaded."""
    lines = [
        "## 📦 Higgs Audio v3 Model Components",
        "",
        "Higgs Studio pulls two pieces from HuggingFace:",
        "",
    ]
    total = 0.0
    for m in MODELS_TO_DOWNLOAD:
        total += m["size_gb"]
        tag = "Required" if m.get("required", True) else "Optional"
        lines += [
            f"### {m['name']} ({tag})",
            f"- **HuggingFace:** [{m['repo_id']}](https://huggingface.co/{m['repo_id']})",
            f"- **Size:** ~{m['size_gb']} GB",
            f"- **Description:** {m['description']}",
            "",
        ]
    lines += [
        "---",
        f"**Total Download Size:** ~{total:.1f} GB",
        "",
        "**Requirements:**",
        "- Stable internet connection",
        "- ~12 GB free disk space recommended",
        "- Downloads resume if interrupted",
        "",
        "> The Higgs weights are the transformers-ported build "
        "(`multimodalart/higgs-audio-v3-tts-4b-transformers`), which is what "
        "exposes `generate_speech()` to plain 🤗 transformers. Boson's raw "
        "`bosonai/higgs-tts-3-4b` repo is for vLLM-Omni / SGLang-Omni serving.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_model_info_text())
    print("\n" + "=" * 50 + "\n")
    ok, msg = check_huggingface_hub()
    print(f"HuggingFace Hub: {msg}")
