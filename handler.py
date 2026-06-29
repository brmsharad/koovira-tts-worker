"""
Koovira OWN Indic TTS worker (RunPod serverless, scale-to-zero).

Model: ai4bharat/indic-parler-tts — 21 languages incl. most Indian regional languages
(Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Malayalam, Marathi, Manipuri,
Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu, English, ...). Language is inferred
from the SCRIPT of the input text (Koovira's KvTranslationEngine already produces the right
script), with an optional natural-voice "description" prompt for style.

Contract (matches classs/KvTts::synthViaRunpod):
  POST /runsync  {"input": {"text": "<utterance>", "description": "<optional voice style>"}}
  ->            {"output": {"audio_base64": "<wav b64>", "format": "wav", "sample_rate": 24000}}

Owns the stack — no external paid TTS service. Free model, pay only for GPU seconds while running.
"""
import base64
import io

import runpod
import soundfile as sf
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
MODEL_ID = "ai4bharat/indic-parler-tts"

_model = None
_tok = None
_desc_tok = None

DEFAULT_DESC = ("A clear, warm, natural-sounding voice speaks at a moderate, expressive pace "
                "with high audio quality and very little background noise.")


def _load():
    global _model, _tok, _desc_tok
    if _model is None:
        _model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        _desc_tok = AutoTokenizer.from_pretrained(_model.config.text_encoder._name_or_path)
    return _model


def handler(event):
    inp = (event or {}).get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "no text"}
    desc = (inp.get("description") or DEFAULT_DESC).strip()
    text = text[:1500]

    m = _load()
    d = _desc_tok(desc, return_tensors="pt").to(DEVICE)
    p = _tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        audio = m.generate(
            input_ids=d.input_ids, attention_mask=d.attention_mask,
            prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
        )
    arr = audio.to(torch.float32).cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, arr, m.config.sampling_rate, format="WAV")
    buf.seek(0)
    return {
        "audio_base64": base64.b64encode(buf.read()).decode("ascii"),
        "format": "wav",
        "sample_rate": int(m.config.sampling_rate),
        "language": inp.get("language_id") or inp.get("language") or "",
    }


runpod.serverless.start({"handler": handler})
