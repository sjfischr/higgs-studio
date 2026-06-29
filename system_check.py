"""
System status checking for Higgs Studio.

Checks CUDA availability/GPU info and whether the Higgs weights (and optional
Moonshine ASR) are present on disk. Mirrors the BeatBunny pattern but keeps the
file checks robust: it looks for a config + at least one safetensors shard
rather than hard-coding exact shard filenames (which vary by upload).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CudaStatus:
    available: bool
    device_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def status_emoji(self) -> str:
        return "✅" if self.available else "❌"

    @property
    def summary(self) -> str:
        if self.available:
            return f"{self.device_name} ({self.vram_total_gb:.1f} GB VRAM)"
        return self.error_message or "CUDA not available"


@dataclass
class ModelComponentStatus:
    name: str
    repo_id: str
    local_path: str
    expected_size_gb: float
    required: bool = True
    downloaded: bool = False
    size_on_disk_gb: float = 0.0
    detail: str = ""

    @property
    def status_emoji(self) -> str:
        if self.downloaded:
            return "✅"
        if self.size_on_disk_gb > 0.01:
            return "⚠️"  # partial
        return "❌"

    @property
    def summary(self) -> str:
        if self.downloaded:
            return f"Ready ({self.size_on_disk_gb:.2f} GB)"
        if self.size_on_disk_gb > 0.01:
            return f"Incomplete ({self.detail or 'partial files'})"
        return "Not downloaded"


@dataclass
class ModelStatus:
    components: Dict[str, ModelComponentStatus]
    model_dir: str

    @property
    def all_ready(self) -> bool:
        return all(c.downloaded for c in self.components.values() if c.required)

    @property
    def status_emoji(self) -> str:
        return "✅" if self.all_ready else "❌"

    @property
    def total_size_gb(self) -> float:
        return sum(c.expected_size_gb for c in self.components.values())

    @property
    def downloaded_size_gb(self) -> float:
        return sum(c.size_on_disk_gb for c in self.components.values())


# Components mirror MODELS_TO_DOWNLOAD. Checks are duck-typed: we want a config
# and at least one weights shard, not a fixed filename list.
MODEL_COMPONENTS = {
    "Higgs-Audio-v3-TTS-4B": {
        "repo_id": "multimodalart/higgs-audio-v3-tts-4b-transformers",
        "expected_size_gb": 9.0,
        "required": True,
        "subfolder": "higgs-audio-v3-tts-4b",
        "needs_config": True,
        "needs_weights": True,
    },
    "Moonshine-Base-ASR": {
        "repo_id": "UsefulSensors/moonshine-base",
        "expected_size_gb": 0.4,
        "required": False,
        "subfolder": "moonshine-base",
        "needs_config": True,
        "needs_weights": True,
    },
}


def check_cuda_status() -> CudaStatus:
    try:
        import torch
    except ImportError:
        return CudaStatus(False, error_message="PyTorch not installed. Run: pip install torch")

    if not torch.cuda.is_available():
        return CudaStatus(
            False,
            error_message="CUDA not available. Install CUDA-enabled PyTorch from pytorch.org",
        )

    try:
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024 ** 3)
        vram_free = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3)
        return CudaStatus(
            available=True,
            device_name=torch.cuda.get_device_name(0),
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            cuda_version=torch.version.cuda,
        )
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return CudaStatus(False, error_message=f"Error checking CUDA: {e}")


def get_folder_size_gb(folder_path: str) -> float:
    total = 0
    folder = Path(folder_path)
    if folder.exists():
        for f in folder.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 3)


def _has_config(path: str) -> bool:
    return any(
        os.path.exists(os.path.join(path, name))
        for name in ("config.json", "preprocessor_config.json")
    )


def _has_weights(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return any(p.rglob("*.safetensors")) or any(p.rglob("*.bin"))


def check_model_component(model_dir: str, name: str, info: dict) -> ModelComponentStatus:
    sub = info["subfolder"]
    local_path = os.path.join(model_dir, sub) if sub else model_dir

    status = ModelComponentStatus(
        name=name,
        repo_id=info["repo_id"],
        local_path=local_path,
        expected_size_gb=info["expected_size_gb"],
        required=info["required"],
    )

    has_cfg = _has_config(local_path) if info.get("needs_config") else True
    has_w = _has_weights(local_path) if info.get("needs_weights") else True
    status.size_on_disk_gb = get_folder_size_gb(local_path)
    status.downloaded = has_cfg and has_w

    if not status.downloaded:
        missing = []
        if info.get("needs_config") and not has_cfg:
            missing.append("config")
        if info.get("needs_weights") and not has_w:
            missing.append("weights")
        status.detail = "missing " + ", ".join(missing) if missing else ""

    return status


def check_model_status(model_dir: str) -> ModelStatus:
    components = {
        name: check_model_component(model_dir, name, info)
        for name, info in MODEL_COMPONENTS.items()
    }
    return ModelStatus(components=components, model_dir=model_dir)


def get_system_status_text(model_dir: str) -> str:
    cuda = check_cuda_status()
    models = check_model_status(model_dir)

    lines = [
        "## 🖥️ System Status",
        "",
        "### GPU (CUDA)",
        f"{cuda.status_emoji} **Status:** {'Available' if cuda.available else 'Not Available'}",
    ]
    if cuda.available:
        lines += [
            f"- **Device:** {cuda.device_name}",
            f"- **VRAM:** {cuda.vram_total_gb:.1f} GB total, {cuda.vram_free_gb:.1f} GB free",
            f"- **CUDA Version:** {cuda.cuda_version}",
        ]
    else:
        lines.append(f"- **Error:** {cuda.error_message}")

    lines += [
        "",
        "### Models",
        f"{models.status_emoji} **Overall:** "
        f"{'All required models ready!' if models.all_ready else 'Required models missing'}",
        f"- **Location:** `{models.model_dir}`",
        f"- **Total Expected:** ~{models.total_size_gb:.1f} GB",
        f"- **On Disk:** {models.downloaded_size_gb:.2f} GB",
        "",
    ]
    for name, comp in models.components.items():
        req = "(Required)" if comp.required else "(Optional)"
        lines += [
            f"#### {comp.status_emoji} {name} {req}",
            f"- **Status:** {comp.summary}",
            f"- **HuggingFace:** `{comp.repo_id}`",
            "",
        ]
    return "\n".join(lines)


def is_ready_to_generate(model_dir: str) -> tuple[bool, str]:
    cuda = check_cuda_status()
    if not cuda.available:
        return False, f"❌ CUDA not available: {cuda.error_message}"

    models = check_model_status(model_dir)
    if not models.all_ready:
        missing = [n for n, c in models.components.items() if c.required and not c.downloaded]
        return False, f"❌ Missing models: {', '.join(missing)}. Go to the Setup tab to download."

    return True, "✅ System ready — synthesize away."


if __name__ == "__main__":
    print(get_system_status_text("./models"))
