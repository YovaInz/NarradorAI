"""
Módulo TTS — SpeechT5 con fallback a pyttsx3
"""
from __future__ import annotations
import os, tempfile
import numpy as np
from typing import Optional, List

_tts_processor = None
_tts_model     = None
_tts_vocoder   = None
_speaker_embed = None
SAMPLE_RATE = 16000

def _load_speecht5():
    global _tts_processor, _tts_model, _tts_vocoder, _speaker_embed
    if _tts_model is not None:
        return True
    try:
        import torch
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        print("[TTSModule] Cargando SpeechT5…")
        _tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        _tts_model     = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        _tts_vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        _tts_model.eval()
        _tts_vocoder.eval()

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        _speaker_embed = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        print("[TTSModule] SpeechT5 listo ✓")
        return True
    except Exception as e:
        print(f"[TTSModule] SpeechT5 no disponible: {e}")
        return False


class TTSModule:
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        if not text or len(text.strip()) < 3:
            return None

        # Intento 1: SpeechT5
        result = self._synthesize_speecht5(text)
        if result is not None:
            return result

        # Intento 2: pyttsx3 (guarda WAV directamente)
        return self._synthesize_pyttsx3(text)

    def _synthesize_speecht5(self, text: str) -> Optional[np.ndarray]:
        try:
            if not _load_speecht5():
                return None
            import torch

            clean = self._clean_text(text)
            inputs = _tts_processor(text=clean, return_tensors="pt")
            with torch.no_grad():
                speech = _tts_model.generate_speech(
                    inputs["input_ids"], _speaker_embed, vocoder=_tts_vocoder
                )
            return speech.numpy()
        except Exception as e:
            print(f"[TTSModule] SpeechT5 error: {e}")
            return None

    def _synthesize_pyttsx3(self, text: str) -> Optional[np.ndarray]:
        """Fallback: pyttsx3 guarda un WAV temporal y lo leemos como numpy."""
        try:
            import pyttsx3, soundfile as sf
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 0.9)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()

            data, sr = sf.read(tmp.name)
            os.unlink(tmp.name)
            return data.astype(np.float32)
        except Exception as e:
            print(f"[TTSModule] pyttsx3 error: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        import re
        clean = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) > 200:
            clean = re.split(r'[.!?]', clean)[0].strip() + "."
        return clean or "The action continues."

    def merge_chunks(self, audio_chunks: List[np.ndarray]) -> Optional[str]:
        if not audio_chunks:
            return None
        try:
            import soundfile as sf
            silence  = np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32)
            parts    = []
            for i, chunk in enumerate(audio_chunks):
                if chunk is not None and len(chunk) > 0:
                    parts.append(chunk.astype(np.float32))
                    if i < len(audio_chunks) - 1:
                        parts.append(silence)
            if not parts:
                return None
            combined = np.concatenate(parts)
            mx = np.max(np.abs(combined))
            if mx > 0:
                combined = combined / mx * 0.9
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, combined, SAMPLE_RATE)
            return tmp.name
        except Exception as e:
            print(f"[TTSModule] merge error: {e}")
            return None
