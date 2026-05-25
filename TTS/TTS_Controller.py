"""
TTS/TTS_Controller.py
---------------------
Mixin que añade toda la lógica de Text-To-Speech a MainWindow.
"""

import queue
import wave
import tempfile
import os

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class TTSLoader(QThread):
    ready  = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            from transformers import pipeline
            tts = pipeline("text-to-speech", model="facebook/mms-tts-spa")
            self.ready.emit(tts)
        except Exception as e:
            self.failed.emit(str(e))


class TTSAudioThread(QThread):
    def __init__(self, tts_pipeline, speed_factor: float = 1.4):
        super().__init__()
        self.tts_pipeline = tts_pipeline
        self.speed_factor = speed_factor
        self._queue: queue.Queue = queue.Queue(maxsize=1)

    def speak(self, text: str):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self):
        self._queue.put(None)

    def run(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._synthesize_and_play(text)

    def _synthesize_and_play(self, text: str):
        tmp_path = None
        try:
            output = self.tts_pipeline(text)
            audio  = np.array(output["audio"])
            if audio.ndim > 1:
                audio = audio.squeeze()
            audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val

            sr          = int(output["sampling_rate"])
            playback_sr = int(sr * self.speed_factor)

            try:
                import sounddevice as sd
                sd.play(audio, samplerate=playback_sr, blocking=True)
                return
            except Exception:
                pass

            pcm = (audio * 32767).astype(np.int16)
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(playback_sr)
                wf.writeframes(pcm.tobytes())

            try:
                import winsound
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            except ImportError:
                os.system(f'aplay "{tmp_path}" 2>/dev/null || afplay "{tmp_path}" 2>/dev/null')

        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"[TTS] No se pudo borrar WAV temporal: {e}")


class TTS_Controller:
    """Mixin de TTS para MainWindow."""

    def _init_tts_state(self):
        self.tts_pipeline  = None
        self.tts_enabled   = True
        self._audio_thread = None

    def _load_tts(self):
        self.status_label.setText("Cargando modelo TTS… (esto puede tomar un momento)")
        self._tts_loader = TTSLoader()
        self._tts_loader.ready.connect(self._on_tts_ready)
        self._tts_loader.failed.connect(self._on_tts_failed)
        self._tts_loader.start()

    def _on_tts_ready(self, pipeline):
        self.tts_pipeline  = pipeline
        self._audio_thread = TTSAudioThread(pipeline, speed_factor=self.spin_tts_speed.value())
        self._audio_thread.start()
        self.status_label.setText("Listo  ·  Narración de voz activada ✔")

    def _on_tts_failed(self, err: str):
        self.tts_pipeline = None
        self.tts_enabled  = False
        self.check_tts.setChecked(False)
        self.check_tts.setEnabled(False)
        self.status_label.setText(f"TTS no disponible: {err}")

    def _speak(self, text: str):
        if not self.tts_enabled or self._audio_thread is None:
            return
        self._audio_thread.speak(text)

    def _on_tts_toggle(self, state: int):
        self.tts_enabled = bool(state)

    def _on_speed_changed(self, value: float):
        if self._audio_thread is not None:
            self._audio_thread.speed_factor = value

    def _stop_tts(self):
        if self._audio_thread and self._audio_thread.isRunning():
            self._audio_thread.stop()
            self._audio_thread.wait(2000)