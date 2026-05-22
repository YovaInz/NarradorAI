import cv2
import base64
import requests
import json
import os
from google import genai
from google.genai import types
from PyQt6.QtCore import QThread, pyqtSignal


class VideoAnalyzerSignals(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)  # Contains path to the resulting JSON
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path,
        interval=2.0,
        api_type="ollama",
        ollama_url="http://localhost:11434",
    ):
        super().__init__()
        self.video_path = video_path
        self.interval = interval
        self.api_type = api_type
        self.ollama_url = ollama_url

        self.api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAW1BT-rSAII6FqCGo7b5ArnENZBdpsMdQ")

        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

        self.json_path = f"{os.path.splitext(self.video_path)[0]}_comments.json"

    def run(self):
        try:
            if os.path.exists(self.json_path):
                self.log.emit("Comments JSON already exists. Skipping analysis.")
                self.finished.emit(self.json_path)
                return

            self.log.emit(f"Starting video analysis using {self.api_type}...")

            self.model = None
            if self.api_type == "gemini":
                if self.api_key == "":
                    self.error.emit(
                        "Gemini API key is not set. Please provide an API key."
                    )
                    return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Could not open video file.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.log.emit(f"Video stats: FPS={fps}, Total Frames={total_frames}")

            frame_interval = int(fps * self.interval)
            current_frame = 0

            comments = []

            prompt = (
                "Eres un comentarista de IA enérgico y creativo para el juego Valorant. "
                "Analiza esta escena de juego de Valorant. Si hay alguna acción destacable, un momento divertido "
                "o algo interesante (como una kill, el uso de una habilidad definitiva, un momento de tensión, un spike, "
                "daño recibido por el jugador, disparos, la aparición de un enemigo, etc.), "
                "escribe un comentario breve y atractivo de máximo 35 palabras, como si fueras un comentarista en tiempo real. "
                "No utilices emojis y utiliza un formato de texto plano"
                "Si no ocurre nada especial, responde simplemente con la palabra 'SKIP'."
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame % frame_interval == 0:
                    timestamp = current_frame / fps

                    # Convert to jpg buffer
                    _, buffer = cv2.imencode(".jpg", frame)

                    self.log.emit(f"Analyzing frame at {timestamp:.2f}s...")

                    if self.api_type == "gemini":
                        image_part = types.Part.from_bytes(
                            data=buffer.tobytes(),
                            mime_type="image/jpeg",
                        )
                        comment = self._analyze_frame_with_gemini(prompt, image_part)
                    else:
                        frame_base64 = base64.b64encode(buffer).decode("utf-8")
                        comment = self._analyze_frame_with_ollama(frame_base64, prompt)

                    if (
                        comment
                        and comment.strip().upper() != "SKIP"
                        and "SKIP" not in comment.strip().upper()
                    ):
                        # Filter out non-notable frames
                        comments.append(
                            {"timestamp": timestamp, "comment": comment.strip()}
                        )
                        self.log.emit(f"Comment generated: {comment.strip()}")

                current_frame += 1
                progress_pct = int((current_frame / total_frames) * 100)
                self.progress.emit(progress_pct)

            cap.release()

            # Save to JSON
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(comments, f, indent=4, ensure_ascii=False)

            self.log.emit("Analysis finished.")
            self.finished.emit(self.json_path)

        except Exception as e:
            self.error.emit(str(e))

    def _analyze_frame_with_gemini(self, prompt, image_part):
        try:
            if self.client is None:
                return "SKIP"

            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[prompt, image_part]
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "SKIP"

    def _analyze_frame_with_ollama(self, base64_image, prompt):
        payload = {
            "model": "gemma4:e4b",
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "SKIP")
        except Exception as e:
            print(f"Ollama API error: {e}")
            return "SKIP"
