"""
🎮 AI Game Commentator — Sistema de Comentarios Automáticos para Videojuegos
Arquitectura: Visión (BLIP) + NLP (GPT-2) + TTS (SpeechT5) + Heurística de Cambio de Escena
"""

import gradio as gr
import os, sys, tempfile, time, json
import numpy as np
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Importaciones de módulos propios
# ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from modules.vision_module import VisionModule
from modules.narration_module import NarrationModule
from modules.tts_module import TTSModule
from modules.control_module import ControlModule
from modules.capture_module import CaptureModule

# ─── Instancias globales (se cargan lazy al primer uso) ─────
vision   = VisionModule()
narrator = NarrationModule()
tts_mod  = TTSModule()
control  = ControlModule()
capture  = CaptureModule()

# ────────────────────────────────────────────────────────────
# Pipeline principal
# ────────────────────────────────────────────────────────────

def process_video(video_path: str, sample_rate: int, sensitivity: float,
                  voice_enabled: bool, commentary_style: str,
                  progress=gr.Progress(track_tqdm=True)):
    """Procesa un video completo y genera comentarios con voz."""

    if not video_path:
        return None, "⚠️ Sube un video para comenzar.", None, "{}"

    log_lines  = []
    audio_path = None
    all_frames_info = []

    try:
        # 1. CAPTURA ─ extraer frames
        progress(0.05, "📽️ Extrayendo frames del video…")
        frames = capture.extract_frames(video_path, sample_rate=sample_rate)
        log_lines.append(f"✅ {len(frames)} frames extraídos (cada {sample_rate}s)")

        if not frames:
            return None, "❌ No se pudieron extraer frames del video.", None, "{}"

        # 2. PROCESAMIENTO por frame con heurística
        control.reset()
        comments = []
        all_audio_chunks = []

        for i, (frame_img, timestamp) in enumerate(
                progress.tqdm(frames, desc="🔍 Analizando escenas")):

            # 2a. VISIÓN — BLIP captioning
            caption = vision.caption_frame(frame_img)
            scene_label = vision.classify_scene(caption)

            # 2b. HEURÍSTICA — ¿comentar ahora?
            should_comment, reason = control.should_comment(
                scene_label, caption, sensitivity=sensitivity
            )

            frame_info = {
                "time": f"{timestamp:.1f}s",
                "caption": caption,
                "scene": scene_label,
                "commented": should_comment,
                "reason": reason,
            }

            if should_comment:
                # 2c. NLP — generar comentario creativo
                comment = narrator.generate_comment(
                    caption, scene_label, style=commentary_style
                )

                # 2d. TTS — convertir a voz (opcional)
                if voice_enabled:
                    audio_chunk = tts_mod.synthesize(comment)
                    if audio_chunk is not None:
                        all_audio_chunks.append(audio_chunk)

                frame_info["comment"] = comment
                comments.append({
                    "time": f"{timestamp:.1f}s",
                    "scene": scene_label,
                    "comment": comment
                })
                log_lines.append(
                    f"[{timestamp:.1f}s] 🎙️ {scene_label} → {comment[:80]}…"
                    if len(comment) > 80 else
                    f"[{timestamp:.1f}s] 🎙️ {scene_label} → {comment}"
                )
            else:
                frame_info["comment"] = ""
                log_lines.append(
                    f"[{timestamp:.1f}s] ⏭️ Sin cambio significativo ({reason})"
                )

            all_frames_info.append(frame_info)

        # 3. COMBINAR AUDIO
        if voice_enabled and all_audio_chunks:
            progress(0.9, "🔊 Ensamblando audio…")
            audio_path = tts_mod.merge_chunks(all_audio_chunks)

        # 4. RESUMEN
        total_comments = len(comments)
        log_lines.append(f"\n{'─'*50}")
        log_lines.append(f"📊 Resumen: {total_comments} comentarios generados en {len(frames)} frames analizados")

        log_text = "\n".join(log_lines)
        stats_json = json.dumps({
            "frames_total":    len(frames),
            "frames_comentados": total_comments,
            "eficiencia":      f"{(total_comments/max(len(frames),1))*100:.1f}%",
            "detalles":        all_frames_info
        }, ensure_ascii=False, indent=2)

        return log_text, format_comments_html(comments), audio_path, stats_json

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"❌ Error: {e}\n\n{tb}", "Error durante el procesamiento", None, "{}"


def format_comments_html(comments: list) -> str:
    if not comments:
        return "Sin comentarios generados."

    scene_icons = {
        "Combate": "⚔️", "Exploración": "🗺️", "Menú/UI": "📋",
        "Cinemática": "🎬", "Personaje con poca vida": "❤️",
        "Victoria/Derrota": "🏆", "Descanso/Quietud": "💤",
        "Acción intensa": "🔥", "General": "🎮"
    }

    html_parts = []
    for c in comments:
        icon = scene_icons.get(c["scene"], "🎮")
        html_parts.append(
            f"<div style='margin-bottom:12px; padding:10px; "
            f"border-left:3px solid #4CAF50; background:#1a1a2e;'>"
            f"<span style='color:#aaa; font-size:12px;'>{c['time']} · {icon} {c['scene']}</span><br>"
            f"<span style='color:#e0e0e0; font-size:14px;'>🎙️ {c['comment']}</span>"
            f"</div>"
        )
    return "".join(html_parts)


# ────────────────────────────────────────────────────────────
# Interfaz Gradio
# ────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="🎮 AI Game Commentator",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=CSS
) as demo:

    gr.HTML("""
    <div id='title'>
      <h1>🎮 AI Game Commentator</h1>
      <p style='color:gray'>
        Sistema multimodal: Visión (BLIP) · NLP (GPT-2) · TTS (SpeechT5) · Heurística de escena
      </p>
    </div>
    """)

    with gr.Row():
        # ── Panel izquierdo: controles ──────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuración")

            video_input = gr.Video(
                label="📁 Video del juego (.mp4)",
                sources=["upload"],
            )

            sample_rate = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="⏱️ Analizar cada N segundos",
                info="Menor = más frames analizados (más lento)"
            )

            sensitivity = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                label="🎯 Sensibilidad al cambio de escena",
                info="Mayor = comenta más frecuentemente"
            )

            commentary_style = gr.Radio(
                choices=["épico", "casual", "técnico", "humorístico"],
                value="épico",
                label="🎭 Estilo del comentarista"
            )

            voice_enabled = gr.Checkbox(
                value=True,
                label="🔊 Activar síntesis de voz (TTS)"
            )

            btn_analyze = gr.Button(
                "🚀 Analizar Video",
                variant="primary",
                size="lg"
            )

            gr.Markdown("---")
            gr.Markdown("""
            **Modelos Hugging Face utilizados:**
            - 🔍 `Salesforce/blip-image-captioning-base` (Visión)
            - 📝 `gpt2` (Generación de texto NLP)
            - 🔊 `microsoft/speecht5_tts` (Síntesis de voz)
            """)

        # ── Panel derecho: resultados ───────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Resultados")

            with gr.Tabs():
                with gr.Tab("🎙️ Comentarios"):
                    comments_output = gr.HTML(
                        label="Comentarios generados",
                        value="<p style='color:gray'>Los comentarios aparecerán aquí…</p>"
                    )

                with gr.Tab("📋 Log de análisis"):
                    log_output = gr.Textbox(
                        label="Log detallado",
                        lines=20,
                        max_lines=30,
                        placeholder="El registro de análisis aparecerá aquí…"
                    )

                with gr.Tab("📈 Estadísticas JSON"):
                    stats_output = gr.Code(
                        label="Datos de análisis (JSON)",
                        language="json",
                        lines=15
                    )

            audio_output = gr.Audio(
                label="🔊 Narración de la jugada",
                type="filepath",
                interactive=False
            )

    # ── Ejemplos de video ───────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 💡 Ejemplo de uso")
    gr.Markdown(
        "Sube cualquier clip de videojuego (.mp4). "
        "El sistema detectará automáticamente cambios de escena "
        "(combate, exploración, menús, etc.) y generará comentarios en español."
    )

    # ── Conexión del botón ──────────────────────────────────
    btn_analyze.click(
        fn=process_video,
        inputs=[
            video_input, sample_rate, sensitivity,
            voice_enabled, commentary_style
        ],
        outputs=[log_output, comments_output, audio_output, stats_output],
        api_name="analyze"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
