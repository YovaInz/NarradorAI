import json
import queue
import wave
import tempfile
import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
    QTextEdit, QProgressBar, QMessageBox, QSlider, QComboBox, QCheckBox
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QThread, pyqtSignal
from analyzer import VideoAnalyzerSignals


# ---------------------------------------------------------------------------
# TTS Loader — carga el modelo en segundo plano al arrancar
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TTSAudioThread — hilo persistente que consume una cola de textos.
# Solo hay UNA instancia durante toda la vida de la app.
# Garantiza que nunca se solapen dos reproducciones.
# ---------------------------------------------------------------------------
class TTSAudioThread(QThread):
    """
    Espera textos en self._queue. Cuando llega uno:
      1. Sintetiza con el pipeline TTS.
      2. Escribe un WAV temporal.
      3. Lo reproduce con winsound (bloqueante → termina antes de procesar el siguiente).
      4. Borra el WAV.

    Sentinel: poner None en la cola detiene el hilo limpiamente.
    Señal 'skip': si ya hay un texto esperando, descartar el actual (el video
    avanzó demasiado rápido) — así nunca se acumula cola de audios viejos.
    """

    def __init__(self, tts_pipeline):
        super().__init__()
        self.tts_pipeline = tts_pipeline
        self._queue: queue.Queue = queue.Queue(maxsize=1)  # máx 1 pendiente

    def speak(self, text: str):
        """Llamado desde el hilo principal. Descarta el anterior si no empezó."""
        # Vaciar la cola antes de meter el nuevo (nunca acumular)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # ya tiene uno encolado, ignorar

    def stop(self):
        self._queue.put(None)  # sentinel

    def run(self):
        import winsound  # Windows; si no existe, usamos el fallback
        while True:
            text = self._queue.get()      # bloquea hasta que llegue algo
            if text is None:              # sentinel → salir
                break
            self._synthesize_and_play(text)

    def _synthesize_and_play(self, text: str):
        tmp_path = None
        try:
            # 1. Sintetizar
            output = self.tts_pipeline(text)
            audio = np.array(output["audio"])
            if audio.ndim > 1:
                audio = audio.squeeze()
            audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = int(output["sampling_rate"])

            # 2. Intentar sounddevice (más limpio, no necesita archivo)
            try:
                import sounddevice as sd
                sd.play(audio, samplerate=sr, blocking=True)
                return
            except Exception:
                pass  # fallback a WAV

            # 3. Fallback WAV → winsound
            pcm = (audio * 32767).astype(np.int16)
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # cerrar el descriptor antes de abrir con wave

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())

            try:
                import winsound
                # SND_FILENAME es bloqueante: no retorna hasta que termina
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            except ImportError:
                os.system(f'aplay "{tmp_path}" 2>/dev/null || afplay "{tmp_path}" 2>/dev/null')

        except Exception as e:
            print(f"[TTS] Error: {e}")

        finally:
            # Borrar el WAV solo DESPUÉS de que winsound haya terminado
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"[TTS] No se pudo borrar WAV temporal: {e}")


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrador IA - Automated Valorant Commentary")
        self.resize(1000, 700)

        self.analyzer_thread = None
        self.comments_data        = []
        self.last_shown_comment_index = -1

        # TTS
        self.tts_pipeline   = None
        self.tts_enabled    = True
        self._audio_thread  = None   # TTSAudioThread — arranca cuando el modelo carga

        self._init_ui()
        self._load_tts()

    # ------------------------------------------------------------------
    # Carga del modelo TTS
    # ------------------------------------------------------------------
    def _load_tts(self):
        self.status_label.setText("Cargando modelo TTS… (esto puede tomar un momento)")
        self.tts_loader = TTSLoader()
        self.tts_loader.ready.connect(self._on_tts_ready)
        self.tts_loader.failed.connect(self._on_tts_failed)
        self.tts_loader.start()

    def _on_tts_ready(self, pipeline):
        self.tts_pipeline  = pipeline
        self._audio_thread = TTSAudioThread(pipeline)
        self._audio_thread.start()
        self.status_label.setText("Listo  ·  Narración de voz activada ✔")

    def _on_tts_failed(self, err):
        self.tts_pipeline = None
        self.tts_enabled  = False
        self.check_tts.setChecked(False)
        self.check_tts.setEnabled(False)
        self.status_label.setText(f"TTS no disponible: {err}")

    def closeEvent(self, event):
        """Detener el hilo de audio al cerrar la ventana."""
        if self._audio_thread and self._audio_thread.isRunning():
            self._audio_thread.stop()
            self._audio_thread.wait(2000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        self.video_widget.setMaximumSize(1280, 720)
        left_layout.addWidget(self.video_widget)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        left_layout.addWidget(self.position_slider)

        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.position_slider.sliderMoved.connect(self.set_position)

        controls_layout = QHBoxLayout()

        self.btn_select_video = QPushButton("Selecciona un video")
        self.btn_select_video.clicked.connect(self.select_video)
        controls_layout.addWidget(self.btn_select_video)

        self.btn_play = QPushButton("Reproducir")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pausar")
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setEnabled(False)
        controls_layout.addWidget(self.btn_pause)

        lbl_rate = QLabel("Tasa de muestra (s):")
        controls_layout.addWidget(lbl_rate)

        self.spin_rate = QDoubleSpinBox()
        self.spin_rate.setRange(0.5, 60.0)
        self.spin_rate.setValue(2.0)
        self.spin_rate.setSingleStep(0.5)
        controls_layout.addWidget(self.spin_rate)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["ollama", "gemini", "openai"])
        controls_layout.addWidget(self.combo_model)

        self.check_force = QCheckBox("Forzar nuevo análisis")
        self.check_force.setToolTip("Ignora el análisis previo y vuelve a procesar el video desde cero.")
        controls_layout.addWidget(self.check_force)

        self.check_tts = QCheckBox("Narración de voz")
        self.check_tts.setChecked(True)
        self.check_tts.setToolTip("Activa/desactiva la síntesis de voz en español.")
        self.check_tts.stateChanged.connect(self._on_tts_toggle)
        controls_layout.addWidget(self.check_tts)

        self.btn_analyze = QPushButton("Comenzar Análisis")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        controls_layout.addWidget(self.btn_analyze)

        left_layout.addLayout(controls_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Listo")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout, stretch=2)

        right_layout = QVBoxLayout()
        lbl_comments = QLabel("Comentarios")
        lbl_comments.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(lbl_comments)

        self.text_comments = QTextEdit()
        self.text_comments.setReadOnly(True)
        right_layout.addWidget(self.text_comments)

        main_layout.addLayout(right_layout, stretch=1)

        self.current_video_path = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_tts_toggle(self, state):
        self.tts_enabled = bool(state)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Video", "", "Archivo de video (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.current_video_path = file_path
            self.status_label.setText(f"Seleccionado: {file_path}")
            self.btn_analyze.setEnabled(True)
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.text_comments.clear()
            self.comments_data = []
            self.last_shown_comment_index = -1
            self.progress_bar.setValue(0)
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(False)

    def start_analysis(self):
        if not self.current_video_path:
            return
        self.media_player.stop()
        self.btn_analyze.setEnabled(False)
        self.btn_select_video.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.text_comments.clear()

        interval        = self.spin_rate.value()
        api_type        = self.combo_model.currentText()
        force_reanalyze = self.check_force.isChecked()

        self.analyzer_thread = VideoAnalyzerSignals(
            self.current_video_path, interval, api_type,
            force_reanalyze=force_reanalyze
        )
        self.analyzer_thread.progress.connect(self.progress_bar.setValue)
        self.analyzer_thread.log.connect(self.status_label.setText)
        self.analyzer_thread.finished.connect(self.analysis_finished)
        self.analyzer_thread.error.connect(self.analysis_error)
        self.analyzer_thread.start()

    def analysis_finished(self, json_path):
        self.btn_analyze.setEnabled(True)
        self.btn_select_video.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.status_label.setText("Análisis completado. Listo para reproducir")
        self.progress_bar.setValue(100)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.comments_data = json.load(f)
            self.comments_data.sort(key=lambda x: x["timestamp"])
            self.last_shown_comment_index = -1
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load comments JSON:\n{e}")

    def analysis_error(self, err_msg):
        self.btn_analyze.setEnabled(True)
        self.btn_select_video.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{err_msg}")
        self.status_label.setText("Analysis failed.")

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.last_shown_comment_index = -1
        self.media_player.setPosition(position)

    # ------------------------------------------------------------------
    # Sincronización de comentarios
    # ------------------------------------------------------------------
    def on_position_changed(self, position_ms):
        self.position_slider.setValue(position_ms)
        if not self.comments_data:
            return

        current_time_s = position_ms / 1000.0

        desired_index = -1
        for i, c in enumerate(self.comments_data):
            if current_time_s >= c["timestamp"]:
                desired_index = i
            else:
                break

        if desired_index < self.last_shown_comment_index:
            # Usuario retrocedió — recargar panel sin TTS
            self.text_comments.clear()
            for i in range(desired_index + 1):
                self._append_comment(self.comments_data[i], speak=False)
            self.last_shown_comment_index = desired_index

        elif desired_index > self.last_shown_comment_index:
            new_comments = []
            for i in range(self.last_shown_comment_index + 1, desired_index + 1):
                self._append_comment(self.comments_data[i], speak=False)
                new_comments.append(self.comments_data[i]["comment"])
            self.last_shown_comment_index = desired_index

            # Solo narrar el más reciente
            if new_comments:
                self._speak(new_comments[-1])

    def _append_comment(self, comment_dict, speak=False):
        time_fmt = self.format_time(comment_dict["timestamp"])
        html = f"<b>[{time_fmt}]</b>: {comment_dict['comment']}<br><br>"
        self.text_comments.append(html)
        if speak:
            self._speak(comment_dict["comment"])

    def _speak(self, text: str):
        if not self.tts_enabled or self._audio_thread is None:
            return
        self._audio_thread.speak(text)   # encola de forma segura

    def format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
