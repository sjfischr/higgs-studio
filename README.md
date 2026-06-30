<div align="center">

# 🍋 Higgs Studio

### Local, expressive text-to-speech with a real expression console.

**Zero-shot voice cloning · 100+ languages · inline control over emotion, style, prosody & sound effects — all running on your own GPU.**

*A clean Gradio front-end for [Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-tts-3-4b) by Boson AI.*

</div>

---

> ⚠️ **Read the [Responsible Use](#-responsible-use--consent) section before you do anything.** Higgs Studio can clone a human voice from a few seconds of audio. That power comes with rules — legal ones and decent-person ones. This tool is for **your own voice and voices you have explicit permission to use.** Nothing else.

---

## Why this exists

Most local TTS front-ends give you a single text box and call it a day, which means the model's best trick — controlling *how* a line is delivered — stays buried. Higgs v3 can shift emotion, style, pitch, speed, pauses, and drop in sound effects mid-sentence, all from inline tags. Higgs Studio surfaces every one of those in the UI so you can actually use them, then runs the whole thing locally with no quotas, no per-character billing, and no audio leaving your machine.

## Features

- 🎙️ **Expression console** — emotion, style, and prosody pickers plus one-click sound-effect and pause buttons that inject the right control tokens for you.
- 🗣️ **Zero-shot voice cloning** — drop in a reference clip; the app auto-transcribes it on CPU to sharpen fidelity (consent required — see below).
- 🌍 **100+ languages** — single-digit WER/CER across 85 production-quality languages.
- 🖥️ **Fully local** — runs on your CUDA GPU. No ZeroGPU quotas, no API keys, no telemetry.
- 🎛️ **Sampling control** — temperature, top-p, top-k, seed, max-tokens, all exposed.
- 💅 **Glassmorphism UI** — tabbed layout, system-status panel, streaming model downloader.

## Quick start

```bash
git clone https://github.com/sjfischr/higgs-studio.git
cd higgs-studio
pip install -r requirements.txt        # install the CUDA build of torch from pytorch.org
python app.py
```

Open the **⚙️ Setup** tab → **Download all models** (~9.4 GB), wait for it to finish, then head to **🎙️ Studio**.

> **A note on the model repo:** Higgs Studio downloads the transformers-ported build
> [`multimodalart/higgs-audio-v3-tts-4b-transformers`](https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers),
> which is what exposes `generate_speech()` to plain 🤗 Transformers. Boson's raw
> `bosonai/higgs-tts-3-4b` weights are built for vLLM-Omni / SGLang-Omni serving and
> won't drive this app directly. Same v3 model, different packaging.

## Hardware

| | |
|---|---|
| **GPU** | NVIDIA with ~12 GB+ VRAM (runs comfortably real-time on an RTX 4090) |
| **Disk** | ~12 GB free for model weights |
| **OS** | Linux / Windows with a CUDA-enabled PyTorch |

## Expression tokens

The pickers in the UI insert these for you, but you can also type them straight into the text box. All tags use `<|category:value|>` syntax and can be placed mid-utterance.

**Placement rules that matter:** put delivery tags (emotion / style / prosody) at the **start** of a turn; put `<|prosody:pause|>` / `<|prosody:long_pause|>` exactly where the pause should land; and **pair every `<|sfx:…|>` with its onomatopoeia immediately after** (e.g. `<|sfx:laughter|>Haha`). Piling many tags head-to-head with no text between them can make generation wander — lead with intent, then let the words carry it.

<details>
<summary><b>Emotion</b> (21)</summary>

`elation` · `amusement` · `enthusiasm` · `determination` · `pride` · `contentment` · `affection` · `relief` · `contemplation` · `confusion` · `surprise` · `awe` · `longing` · `arousal` · `anger` · `fear` · `disgust` · `bitterness` · `sadness` · `shame` · `helplessness`

</details>

<details>
<summary><b>Style</b> (3)</summary>

`singing` · `shouting` · `whispering`

</details>

<details>
<summary><b>Sound effects</b> (9) — pair with onomatopoeia</summary>

| Token | Say right after |
|---|---|
| `sfx:cough` | Ahem |
| `sfx:laughter` | Haha / Hehe |
| `sfx:crying` | Boohoo / Sob |
| `sfx:screaming` | Ahh / Aaah |
| `sfx:burping` | Burp |
| `sfx:humming` | Hmm / Mmm |
| `sfx:sigh` | Uh / Ahh |
| `sfx:sniff` | Sff |
| `sfx:sneeze` | Achoo |

</details>

<details>
<summary><b>Prosody</b> (10)</summary>

`speed_very_slow` (≈0.65×) · `speed_slow` (≈0.85×) · `speed_fast` (≈1.2×) · `speed_very_fast` (≈1.4×) · `pitch_low` (≈−3 st) · `pitch_high` (≈+2.5 st) · `pause` (≈400–700 ms) · `long_pause` (≈700–1500 ms) · `expressive_high` · `expressive_low`

</details>

**Example:**

```
<|emotion:amusement|><|prosody:expressive_high|>Wait, that was kind of hilarious. <|sfx:laughter|>Hehe, I was not ready for that.
```

## 🛡️ Responsible Use & Consent

Higgs Studio synthesizes and clones human voices. **Read this. It's short and it's not optional.**

**Only clone voices you're allowed to clone.**
- ✅ Your own voice.
- ✅ A voice you have **explicit, documented permission** to use.
- ❌ Anyone else. Full stop.

**Never use Higgs Studio to:**
- Impersonate a real person to deceive, defraud, or take action on their behalf.
- Create voice content of someone without their consent — *especially* anything sexual, harassing, or defamatory.
- Generate political or election content designed to mislead.
- Produce robocalls, scam calls, or any communication that pretends to be someone it isn't.
- Clone the voice of a minor, for any reason.

**The law is real and getting realer.** Voice is now a protected right in a growing number of places. Tennessee's **ELVIS Act** makes unauthorized voice cloning actionable and explicitly reaches the *distribution* of cloning tools; the federal **NO FAKES Act** (advancing through Congress as of mid-2026) would create liability for producing *or distributing* unauthorized digital voice replicas. "I only ran the software" is not the safe harbor people assume it is. Know your local law before you generate.

**Disclose AI-generated audio** when you share it, so listeners aren't deceived.

This project ships **no watermarking or identity verification** — it's a local research tool, not a managed platform. That puts the entire burden of responsible use on **you**. If you can't carry that, don't use it.

> Built something cool and consensual with it? That's the point. Built something that hurts someone? That's on you, and the maintainers want no part of it.

## License & attribution

- **Higgs Studio code:** [MIT] © its contributors.
- **Higgs Audio v3 model:** governed by the **Boson Higgs TTS 3 Research and Non-Commercial License.** Personal, non-commercial, self-hosted use only. **Commercial, hosted, or revenue-generating use requires a separate license from [Boson AI](https://www.boson.ai/).** This repo is non-commercial by design.
- **Original Gradio app:** adapted from [`multimodalart/higgs-audio-v3-tts`](https://huggingface.co/spaces/multimodalart/higgs-audio-v3-tts).   
- **Moonshine ASR:** [`UsefulSensors/moonshine-base`](https://huggingface.co/UsefulSensors/moonshine-base), used for reference-clip transcription.

By using this software you agree to the model license and to the responsible-use terms above.

## Acknowledgments

Higgs Audio v3 by **Boson AI** · transformers port by **multimodalart** · ASR by **Useful Sensors**.

---

<div align="center">
<sub>Higgs Studio is an independent, non-commercial wrapper. Not affiliated with or endorsed by Boson AI.</sub>
</div>