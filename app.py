"""
Higgs Studio — a local, enriched Gradio app for Higgs Audio v3 TTS.

Cloned (not formally forked) from multimodalart/higgs-audio-v3-tts and reworked:
  * ZeroGPU / `spaces` plumbing removed — runs directly on your local CUDA GPU.
  * Weights load from a local directory you download via the Setup tab.
  * Expression controls (emotion / style / sfx / pauses) surfaced in the UI,
    so the model's inline control tokens are one click away instead of buried.
  * BeatBunny-style tabbed layout, glassmorphism styling, and system status.

Model: Higgs Audio v3 TTS (4B), transformers-ported build. `generate_speech`
returns a mono 24 kHz float32 waveform.
"""

import os
import logging
import sys

import torch
import torchaudio
import soundfile as sf
import gradio as gr
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MoonshineForConditionalGeneration,
)

from system_check import (
    check_cuda_status,
    get_system_status_text,
    is_ready_to_generate,
)
from model_downloader import (
    download_all_models_generator,
    get_model_info_text,
    cancel_download,
)

load_dotenv()


# --- Configuration ---
class Config:
    BASE_DIR = os.getcwd()
    MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))
    LOG_FILE = os.path.join(LOG_DIR, "higgs_studio.log")

    HIGGS_SUBDIR = "higgs-audio-v3-tts-4b"
    MOONSHINE_SUBDIR = "moonshine-base"

    @classmethod
    def ensure_directories(cls):
        for p in [cls.MODEL_DIR, cls.OUTPUT_DIR, cls.LOG_DIR]:
            os.makedirs(p, exist_ok=True)

    @classmethod
    def higgs_path(cls):
        return os.path.join(cls.MODEL_DIR, cls.HIGGS_SUBDIR)

    @classmethod
    def moonshine_path(cls):
        return os.path.join(cls.MODEL_DIR, cls.MOONSHINE_SUBDIR)


def setup_logging():
    Config.ensure_directories()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.hasHandlers():
        root.handlers.clear()
    fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    logging.info("Logging initialized.")


setup_logging()
logger = logging.getLogger("higgs_studio")


# --- Lazy model loading -------------------------------------------------------
# We do NOT load the 4B model at import time. That way the app boots even before
# the weights are downloaded, and the Setup tab stays usable. The first synth
# call (or the Load button) brings the model in and caches it.
_MODEL = None
_TOKENIZER = None
_SR = 24000  # Higgs v3 emits 24 kHz; confirmed from model.config after load.

_ASR_MODEL = None
_ASR_PROCESSOR = None
ASR_SAMPLE_RATE = 16000


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_higgs():
    """Load and cache the Higgs model + tokenizer from the local model dir."""
    global _MODEL, _TOKENIZER, _SR
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    path = Config.higgs_path()
    if not os.path.isdir(path):
        raise gr.Error(
            "Higgs weights not found. Open the ⚙️ Setup tab and download the model first."
        )

    logger.info(f"Loading Higgs Audio v3 from {path} ...")
    _TOKENIZER = AutoTokenizer.from_pretrained(path)
    _MODEL = (
        AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
        .to(_device())
        .eval()
    )
    _MODEL.get_audio_codec()  # preload the 24 kHz codec
    _SR = getattr(_MODEL.config, "sample_rate", 24000)
    logger.info(f"Higgs loaded. Sample rate = {_SR} Hz.")
    return _MODEL, _TOKENIZER


def load_asr():
    """Load and cache Moonshine ASR (CPU). Prefers local copy, falls back to hub."""
    global _ASR_MODEL, _ASR_PROCESSOR
    if _ASR_MODEL is not None:
        return _ASR_MODEL, _ASR_PROCESSOR

    local = Config.moonshine_path()
    src = local if os.path.isdir(local) else "UsefulSensors/moonshine-base"
    logger.info(f"Loading Moonshine ASR from {src} ...")
    _ASR_PROCESSOR = AutoProcessor.from_pretrained(src)
    _ASR_MODEL = MoonshineForConditionalGeneration.from_pretrained(src).eval()
    return _ASR_MODEL, _ASR_PROCESSOR


# --- Core functions -----------------------------------------------------------
def transcribe(reference_audio):
    """CPU auto-transcription of the reference clip to seed the cloning transcript."""
    if not reference_audio:
        return gr.update()
    try:
        asr_model, asr_proc = load_asr()
    except Exception as e:
        logger.warning(f"ASR unavailable: {e}")
        return gr.update()

    data, sr = sf.read(reference_audio, dtype="float32", always_2d=True)  # [L, C]
    wav = torch.from_numpy(data).mean(dim=1)  # mono [L]
    if sr != ASR_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=ASR_SAMPLE_RATE)
    inputs = asr_proc(wav.numpy(), sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        tokens = asr_model.generate(**inputs)
    return asr_proc.decode(tokens[0], skip_special_tokens=True).strip()


def synthesize(text, reference_audio, reference_text, temperature, top_p, top_k, max_new_tokens, seed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some text to synthesize.")

    model, tokenizer = load_higgs()

    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))

    kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p) if float(top_p) < 1.0 else None,
        top_k=int(top_k) if int(top_k) > 0 else None,
    )

    if reference_audio is not None:
        data, sr = sf.read(reference_audio, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)  # mono [L]
        kwargs["reference_audio"] = wav
        kwargs["reference_sample_rate"] = sr
        if reference_text and reference_text.strip():
            kwargs["reference_text"] = reference_text.strip()

    audio = model.generate_speech(text, tokenizer, **kwargs)
    if audio.numel() == 0:
        raise gr.Error("Generation produced no audio — try again or adjust the text.")
    return (_SR, audio.numpy())


# --- Expression-token helpers -------------------------------------------------
# All tags are from the official Boson docs: https://docs.boson.ai/models/higgs-tts/tags
# Lead-the-turn tags (emotion, style, prosody) go at the START of the text.
# Positional tags (sfx, pause) go exactly WHERE the effect should occur.

# 21 official emotion tags
EMOTION_TAGS = [
    "elation", "amusement", "enthusiasm", "determination",
    "pride", "contentment", "affection", "relief", "contemplation",
    "confusion", "surprise", "awe", "longing", "arousal",
    "anger", "fear", "disgust", "bitterness", "sadness",
    "shame", "helplessness",
]

# 3 official style tags
STYLE_TAGS = ["singing", "shouting", "whispering"]

# 8 lead-the-turn prosody tags (set delivery for the whole turn)
PROSODY_LEAD_TAGS = [
    "expressive_high", "expressive_low",
    "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
    "pitch_low", "pitch_high",
]

# 9 positional sound-effect tokens (vocalized in the speaker's voice)
SFX_TOKENS = {
    "😷 Cough": "<|sfx:cough|>",
    "😂 Laughter": "<|sfx:laughter|>",
    "😢 Crying": "<|sfx:crying|>",
    "😱 Screaming": "<|sfx:screaming|>",
    "🫠 Burping": "<|sfx:burping|>",
    "🎵 Humming": "<|sfx:humming|>",
    "😮‍💨 Sigh": "<|sfx:sigh|>",
    "👃 Sniff": "<|sfx:sniff|>",
    "🤧 Sneeze": "<|sfx:sneeze|>",
}

# 2 positional prosody pause tokens
PAUSE_TOKENS = {
    "⏸️ Pause (~400–700 ms)": "<|prosody:pause|>",
    "⏳ Long pause (~700–1500 ms)": "<|prosody:long_pause|>",
}


def insert_token(current_text, token):
    """Append a positional tag (sfx or pause) at the end of the current text."""
    current_text = current_text or ""
    if current_text and not current_text.endswith(" "):
        current_text += " "
    return current_text + token


def apply_expression(current_text, emotion, style, prosody):
    """Prepend lead-the-turn tags to the utterance; they must appear at the start."""
    current_text = (current_text or "").lstrip()
    prefix = ""
    if emotion:
        prefix += f"<|emotion:{emotion.strip()}|>"
    if style:
        prefix += f"<|style:{style.strip()}|>"
    if prosody:
        prefix += f"<|prosody:{prosody.strip()}|>"
    return prefix + current_text


# --- Theme + CSS (BeatBunny glassmorphism, lemon/amber palette) ---------------
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="yellow",
    neutral_hue="slate",
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
)

CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(-45deg, #1a1208, #2b2410, #3a2e0e, #241c0a);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: #f5f0e6 !important;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.glass-panel {
    background: rgba(255, 240, 200, 0.05) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 220, 130, 0.15) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}
h1, h2, h3 {
    text-shadow: 0 0 10px rgba(255, 210, 120, 0.3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
textarea, input {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 220, 130, 0.12) !important;
    color: #f5f0e6 !important;
}
button.primary {
    background: linear-gradient(90deg, #ffb300, #ff7043) !important;
    border: none !important;
    color: #1a1208 !important;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(255, 179, 0, 0.4) !important;
    transition: all 0.3s ease;
}
button.primary:hover {
    box-shadow: 0 0 25px rgba(255, 179, 0, 0.7) !important;
    transform: translateY(-2px);
}
audio { width: 100%; filter: sepia(40%) hue-rotate(5deg) saturate(160%); }
"""


# --- UI helpers ---------------------------------------------------------------
def refresh_system_status():
    return get_system_status_text(Config.MODEL_DIR)


def get_readiness_banner():
    ready, message = is_ready_to_generate(Config.MODEL_DIR)
    return gr.update(value=message, visible=not ready)


def run_model_download():
    yield "Starting download process...\n"
    for msg in download_all_models_generator(Config.MODEL_DIR):
        yield msg


# --- UI -----------------------------------------------------------------------
with gr.Blocks(title="Higgs Studio", theme=theme, css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🍋 Higgs `Studio`")
    gr.Markdown("_Local Higgs Audio v3 TTS — zero-shot voice cloning + inline expression control._")

    with gr.Tabs():
        # ============ STUDIO TAB ============
        with gr.Tab("🎙️ Studio", id="studio-tab"):
            readiness_banner = gr.Textbox(
                value="", visible=False, interactive=False, show_label=False,
                elem_id="readiness-banner",
            )

            with gr.Row():
                # LEFT: text + expression
                with gr.Column(scale=2, elem_classes="glass-panel"):
                    gr.Markdown("### 1. Text to synthesize")
                    text = gr.Textbox(
                        label="Text",
                        placeholder="Type what you want the voice to say…",
                        lines=6,
                        elem_id="text-box",
                    )

                    gr.Markdown("### 2. Expression")
                    gr.Markdown(
                        "💡 **Lead-the-turn tags** (Emotion, Style, Prosody) are prepended to the text — "
                        "place them at the very start. **Positional tags** (SFX, Pause) go where the effect should occur."
                    )
                    with gr.Row():
                        emotion = gr.Dropdown(
                            label="Emotion", choices=EMOTION_TAGS,
                            value=None, allow_custom_value=True, scale=1,
                        )
                        style = gr.Dropdown(
                            label="Style", choices=STYLE_TAGS,
                            value=None, allow_custom_value=True, scale=1,
                        )
                        prosody = gr.Dropdown(
                            label="Prosody", choices=PROSODY_LEAD_TAGS,
                            value=None, allow_custom_value=True, scale=1,
                        )
                    apply_expr_btn = gr.Button("Apply emotion / style / prosody to start of text", size="sm")

                    gr.Markdown("**Sound effects** (positional — inserted at end, move to reposition):")
                    with gr.Row():
                        sfx_buttons = [gr.Button(label, size="sm") for label in SFX_TOKENS]

                    gr.Markdown("**Pauses** (positional):")
                    with gr.Row():
                        pause_buttons = [gr.Button(label, size="sm") for label in PAUSE_TOKENS]

                    gr.Markdown("### 3. Sampling")
                    with gr.Accordion("Advanced settings", open=False):
                        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                        top_k = gr.Slider(0, 1026, value=50, step=1, label="Top-k (0 = off)")
                        max_new_tokens = gr.Slider(64, 4096, value=2048, step=64, label="Max new tokens")
                        seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                    run = gr.Button("🔊 Generate speech", variant="primary")

                # RIGHT: voice cloning + output
                with gr.Column(scale=2, elem_classes="glass-panel"):
                    gr.Markdown("### Voice cloning (optional)")
                    reference_audio = gr.Audio(
                        label="Reference voice — upload or record a clip", type="filepath",
                    )
                    reference_text = gr.Textbox(
                        label="Reference transcript (auto-filled on upload, improves cloning)",
                        lines=2,
                    )

                    gr.Markdown("### Output")
                    output_audio = gr.Audio(label="Generated speech", type="numpy")

                    gr.Examples(
                        examples=[
                            ["Higgs Audio version three, now running locally.", None, ""],
                            ["<|emotion:amusement|><|prosody:expressive_high|>Wait, that was kind of hilarious. <|sfx:laughter|>Hehe.", None, ""],
                            ["The quick brown fox jumps over the lazy dog.", None, ""],
                        ],
                        inputs=[text, reference_audio, reference_text],
                    )

            # Wiring
            reference_audio.change(
                transcribe, inputs=[reference_audio], outputs=[reference_text], api_name="transcribe"
            )
            apply_expr_btn.click(apply_expression, [text, emotion, style, prosody], text)
            for btn, token in zip(sfx_buttons, SFX_TOKENS.values()):
                btn.click(lambda cur, tok=token: insert_token(cur, tok), [text], text)
            for btn, token in zip(pause_buttons, PAUSE_TOKENS.values()):
                btn.click(lambda cur, tok=token: insert_token(cur, tok), [text], text)
            run.click(
                synthesize,
                inputs=[text, reference_audio, reference_text, temperature, top_p, top_k, max_new_tokens, seed],
                outputs=output_audio,
            )

        # ============ SETUP TAB ============
        with gr.Tab("⚙️ Setup", id="setup-tab"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass-panel"):
                    gr.Markdown("## 📥 Download models")
                    gr.Markdown(get_model_info_text())
                    with gr.Row():
                        download_btn = gr.Button("⬇️ Download all models", variant="primary")
                        cancel_btn = gr.Button("✖️ Cancel", variant="secondary")
                    download_output = gr.Markdown("")

                with gr.Column(scale=1, elem_classes="glass-panel"):
                    gr.Markdown("## 🖥️ System status")
                    status_box = gr.Markdown(get_system_status_text(Config.MODEL_DIR))
                    refresh_btn = gr.Button("🔄 Refresh status", size="sm")

            download_btn.click(run_model_download, outputs=download_output).then(
                refresh_system_status, outputs=status_box
            ).then(get_readiness_banner, outputs=readiness_banner)
            cancel_btn.click(cancel_download)
            refresh_btn.click(refresh_system_status, outputs=status_box).then(
                get_readiness_banner, outputs=readiness_banner
            )

    demo.load(get_readiness_banner, outputs=readiness_banner)


if __name__ == "__main__":
    cuda = check_cuda_status()
    logger.info(f"CUDA: {cuda.summary}")
    demo.queue().launch()
