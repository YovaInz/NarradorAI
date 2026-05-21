"""
Módulo de Captura (#1)
─────────────────────
Extrae frames de un archivo .mp4 usando OpenCV.
Retorna lista de (PIL.Image, timestamp_segundos).
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


class CaptureModule:
    """
    Responsable de abrir el video y extraer frames
    a intervalos regulares para su análisis.
    """

    def extract_frames(
        self,
        video_path: str,
        sample_rate: float = 3.0,
        max_frames: int = 120,
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[Tuple[Image.Image, float]]:
        """
        Extrae frames cada `sample_rate` segundos.

        Parameters
        ----------
        video_path  : ruta al archivo de video (.mp4)
        sample_rate : segundos entre cada frame analizado
        max_frames  : límite de seguridad para videos muy largos
        target_size : tamaño de salida (ancho, alto) en píxeles

        Returns
        -------
        Lista de tuplas (imagen_PIL, timestamp_segundos)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"No se puede abrir el video: {video_path}")

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_f / fps

        # Intervalo en frames
        frame_interval = max(1, int(fps * sample_rate))

        frames: List[Tuple[Image.Image, float]] = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # BGR → RGB → PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb).resize(
                target_size, Image.LANCZOS
            )

            frames.append((pil_img, timestamp))
            frame_idx += frame_interval

        cap.release()
        return frames

    def capture_screen(self) -> Image.Image:
        """
        Captura la pantalla completa (requiere mss).
        Alternativa cuando no hay archivo de video.
        """
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                return Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb
                )
        except ImportError:
            raise ImportError(
                "Instala 'mss' para captura de pantalla: pip install mss"
            )
