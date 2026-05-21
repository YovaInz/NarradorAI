"""
Módulo de Control — Lógica Heurística (#5)
──────────────────────────────────────────
Implementa la lógica de decisión para determinar cuándo comentar.

Estrategia (heurística basada en reglas):
1. Cambio de escena  : si la etiqueta cambia respecto al frame anterior → comentar
2. Tiempo mínimo     : no comentar si no pasaron suficientes segundos
3. Cooldown global   : máximo N comentarios por minuto para no saturar
4. Prioridad urgente : ciertos eventos (vida baja, victoria) tienen prioridad

Esta implementación refleja los principios de búsqueda heurística
vistos en clase: evaluación de estado, umbral de cambio, coste de acción.
"""

from __future__ import annotations
import time
from collections import deque
from typing import Tuple


class ControlModule:
    """
    Motor de decisión heurístico:
    decide frame a frame si el comentarista debe hablar.
    """

    # Escenas que siempre justifican un comentario (alta prioridad)
    HIGH_PRIORITY_SCENES = {
        "Personaje con poca vida",
        "Victoria/Derrota",
        "Acción intensa",
    }

    # Tiempo mínimo (segundos) entre comentarios consecutivos
    MIN_COMMENT_INTERVAL = 5.0

    # Máximo de comentarios en una ventana deslizante (60s)
    MAX_COMMENTS_PER_MINUTE = 8

    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia el estado del controlador al procesar un nuevo video."""
        self._last_scene:     str   = ""
        self._last_caption:   str   = ""
        self._last_comment_t: float = -999.0  # "tiempo" del último comentario
        self._frame_counter:  int   = 0
        self._comment_times:  deque = deque(maxlen=20)  # ventana deslizante

        # Contador de apariciones por escena (evitar comentar lo mismo muchas veces)
        self._scene_counts: dict[str, int] = {}

    def should_comment(
        self,
        scene_label: str,
        caption: str,
        current_time: float = None,
        sensitivity: float = 0.5,
    ) -> Tuple[bool, str]:
        """
        Evalúa si se debe generar un comentario para este frame.

        Parameters
        ----------
        scene_label  : etiqueta de escena detectada por VisionModule
        caption      : descripción textual del frame
        current_time : timestamp del frame en segundos (si None, usa contador)
        sensitivity  : 0.0 = muy restrictivo, 1.0 = comenta casi siempre

        Returns
        -------
        (decision: bool, razón: str)
        """
        self._frame_counter += 1

        # Usar contador como tiempo simulado si no se provee timestamp real
        t = current_time if current_time is not None else float(self._frame_counter)

        # ── Regla 1: Alta prioridad siempre comenta (con cooldown mínimo) ──
        if scene_label in self.HIGH_PRIORITY_SCENES:
            if t - self._last_comment_t >= self.MIN_COMMENT_INTERVAL * 0.5:
                return self._accept(scene_label, caption, t,
                                    "Alta prioridad: " + scene_label)

        # ── Regla 2: Cooldown mínimo entre comentarios ──────────────────────
        min_interval = self.MIN_COMMENT_INTERVAL * (2.0 - sensitivity)
        if t - self._last_comment_t < min_interval:
            return False, f"Cooldown ({t - self._last_comment_t:.1f}s < {min_interval:.1f}s)"

        # ── Regla 3: Límite de comentarios por minuto ───────────────────────
        recent = sum(1 for ct in self._comment_times if t - ct < 60.0)
        max_per_min = int(self.MAX_COMMENTS_PER_MINUTE * sensitivity)
        if recent >= max_per_min:
            return False, f"Límite por minuto ({recent}/{max_per_min})"

        # ── Regla 4: Cambio de escena detectado ─────────────────────────────
        if scene_label != self._last_scene and self._last_scene != "":
            return self._accept(scene_label, caption, t,
                                f"Cambio de escena: {self._last_scene} → {scene_label}")

        # ── Regla 5: Primer frame siempre comenta ───────────────────────────
        if self._frame_counter == 1 or self._last_scene == "":
            return self._accept(scene_label, caption, t, "Primer frame")

        # ── Regla 6: Variación significativa en el caption ──────────────────
        if self._caption_changed_significantly(caption, self._last_caption, sensitivity):
            return self._accept(scene_label, caption, t, "Cambio visual significativo")

        # ── Regla 7: Escena repetida — comentar con frecuencia reducida ─────
        scene_count = self._scene_counts.get(scene_label, 0)
        repeat_interval = 15.0 / max(sensitivity, 0.1)  # más sensible = más frecuente
        if t - self._last_comment_t >= repeat_interval:
            return self._accept(scene_label, caption, t,
                                f"Repetición programada (escena x{scene_count + 1})")

        return False, "Sin cambio significativo"

    def _accept(
        self, scene_label: str, caption: str, t: float, reason: str
    ) -> Tuple[bool, str]:
        """Registra la decisión de comentar y actualiza el estado."""
        self._last_scene     = scene_label
        self._last_caption   = caption
        self._last_comment_t = t
        self._comment_times.append(t)
        self._scene_counts[scene_label] = self._scene_counts.get(scene_label, 0) + 1
        return True, reason

    def _caption_changed_significantly(
        self, new: str, old: str, sensitivity: float
    ) -> bool:
        """
        Mide qué tan diferente es el nuevo caption del anterior.
        Usa la distancia de Jaccard entre conjuntos de palabras.
        """
        if not old or not new:
            return False

        words_new = set(new.lower().split())
        words_old = set(old.lower().split())

        intersection = len(words_new & words_old)
        union        = len(words_new | words_old)

        if union == 0:
            return False

        similarity = intersection / union
        # Umbral: a mayor sensibilidad, menor umbral de "cambio"
        threshold = 0.7 - (sensitivity * 0.4)  # 0.3 a 0.7

        return similarity < threshold
