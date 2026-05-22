import json
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox, 
    QTextEdit, QProgressBar, QMessageBox, QSlider, QComboBox
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt
from analyzer import VideoAnalyzerSignals

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrador IA - Automated Valorant Commentary")
        self.resize(1000, 700)
        
        self.analyzer_thread = None
        self.comments_data = [] # List of {timestamp: float, comment: str}
        self.last_shown_comment_index = -1
        
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Video player and Controls
        left_layout = QVBoxLayout()
        
        # Video area
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
        
        # Controls area
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
        self.combo_model.addItems(["ollama", "gemini"])
        controls_layout.addWidget(self.combo_model)
        
        self.btn_analyze = QPushButton("Comenzar Análisis")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        controls_layout.addWidget(self.btn_analyze)
        
        left_layout.addLayout(controls_layout)
        
        # Progress and Logs
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Listo")
        left_layout.addWidget(self.status_label)
        
        main_layout.addLayout(left_layout, stretch=2)
        
        # Right side: Commentary box
        right_layout = QVBoxLayout()
        lbl_comments = QLabel("Comentarios")
        lbl_comments.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(lbl_comments)
        
        self.text_comments = QTextEdit()
        self.text_comments.setReadOnly(True)
        right_layout.addWidget(self.text_comments)
        
        main_layout.addLayout(right_layout, stretch=1)
        
        self.current_video_path = None

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Video", "", "Archivo de video (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.current_video_path = file_path
            self.status_label.setText(f"Seleccionado: {file_path}")
            self.btn_analyze.setEnabled(True)
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            
            # Reset UI state
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
        
        interval = self.spin_rate.value()
        api_type = self.combo_model.currentText()
        
        self.analyzer_thread = VideoAnalyzerSignals(self.current_video_path, interval, api_type)
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
        
        self.status_label.setText(f"Análisis completado. Listo para reproducir")
        self.progress_bar.setValue(100)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.comments_data = json.load(f)
            # Ensure chronological order
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
        self.media_player.setPosition(position)

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
            self.text_comments.clear()
            for i in range(desired_index + 1):
                c = self.comments_data[i]
                time_formatted = self.format_time(c["timestamp"])
                new_text = f"<b>[{time_formatted}]</b>: {c['comment']}<br><br>"
                self.text_comments.append(new_text)
            self.last_shown_comment_index = desired_index
        elif desired_index > self.last_shown_comment_index:
            for i in range(self.last_shown_comment_index + 1, desired_index + 1):
                c = self.comments_data[i]
                time_formatted = self.format_time(c["timestamp"])
                new_text = f"<b>[{time_formatted}]</b>: {c['comment']}<br><br>"
                self.text_comments.append(new_text)
            self.last_shown_comment_index = desired_index

    def format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
