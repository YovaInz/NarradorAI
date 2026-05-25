from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import QUrl, Qt


class VideoPlayer:
    """
    Mixin de reproducción de video y sincronización de comentarios.
    Se usa junto con QMainWindow y TTSController en herencia múltiple.
    """

    # ------------------------------------------------------------------
    # Inicialización — llamar desde _init_ui de MainWindow
    # ------------------------------------------------------------------
    def _init_video_player(self, left_layout):
        """
        Crea y conecta el widget de video, el slider y el QMediaPlayer.
        Agrega los widgets al layout recibido.
        """
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
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)

        # Estado de comentarios
        self.comments_data            = []
        self.last_shown_comment_index = -1

    # ------------------------------------------------------------------
    # Slots de reproducción
    # ------------------------------------------------------------------
    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def _on_duration_changed(self, duration: int):
        self.position_slider.setRange(0, duration)

    def _on_slider_moved(self, position: int):
        self.last_shown_comment_index = -1
        self.media_player.setPosition(position)

    def load_video(self, file_path: str):
        """Carga un archivo de video y resetea el estado de comentarios."""
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.text_comments.clear()
        self.comments_data            = []
        self.last_shown_comment_index = -1
        self.position_slider.setValue(0)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)

    def load_comments(self, comments: list):
        """Recibe la lista de comentarios ya ordenada desde MainWindow."""
        self.comments_data            = comments
        self.last_shown_comment_index = -1

    # ------------------------------------------------------------------
    # Sincronización de comentarios con el tiempo del video
    # ------------------------------------------------------------------
    def on_position_changed(self, position_ms: int):
        self.position_slider.setValue(position_ms)
        if not self.comments_data:
            return

        current_time_s = position_ms / 1000.0
        desired_index  = -1
        for i, c in enumerate(self.comments_data):
            if current_time_s >= c["timestamp"]:
                desired_index = i
            else:
                break

        if desired_index < self.last_shown_comment_index:
            # El usuario retrocedió — recargar panel sin narrar
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
            if new_comments:
                self._speak(new_comments[-1])  # narrar solo el más reciente

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_comment(self, comment_dict: dict, speak: bool = False):
        time_fmt = self._format_time(comment_dict["timestamp"])
        html     = f"<b>[{time_fmt}]</b>: {comment_dict['comment']}<br><br>"
        self.text_comments.append(html)
        if speak:
            self._speak(comment_dict["comment"])

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
