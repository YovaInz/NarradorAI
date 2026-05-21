# 🎮 AI Game Commentator

> Sistema multimodal de IA para generar comentarios automáticos de videojuegos en tiempo real.

---

## 📐 Arquitectura del Sistema

```
Video (.mp4)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULO 1 — CAPTURA (capture_module.py)                             │
│  OpenCV → extrae frames cada N segundos → lista de (PIL.Image, t)   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ frame + timestamp
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULO 5 — CONTROL HEURÍSTICO (control_module.py)                  │
│  ¿Hay cambio significativo? ¿Cooldown OK? ¿Prioridad alta?          │
│  Reglas: cambio escena · distancia Jaccard · cooldown · prioridad   │
└──────────┬──────────────────────────────────────────────────────────┘
           │ ¿comentar? SÍ
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULO 2 — VISIÓN (vision_module.py) — HuggingFace modelo #1       │
│  Salesforce/blip-image-captioning-base                              │
│  PIL.Image → BLIP Caption (EN) → Classify Scene (reglas)           │
│  "a screenshot showing combat with sword" → "Combate"               │
└──────────┬──────────────────────────────────────────────────────────┘
           │ caption + scene_label
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULO 3 — NARRACIÓN NLP (narration_module.py) — HuggingFace #2   │
│  gpt2                                                               │
│  prompt contextual → GPT-2 sampling → post-proceso → ES             │
│  "¡INCREÍBLE! El jugador está en combate intenso."                  │
└──────────┬──────────────────────────────────────────────────────────┘
           │ comentario en texto
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULO 4 — TTS (tts_module.py) — HuggingFace modelo #3             │
│  microsoft/speecht5_tts + speecht5_hifigan                          │
│  texto → waveform (float32, 16kHz) → WAV                           │
└──────────┬──────────────────────────────────────────────────────────┘
           │ audio WAV
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INTERFAZ GRADIO (app.py)                                           │
│  Video player · Comentarios HTML · Log de análisis · Audio player  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤗 Modelos de Hugging Face Utilizados

| # | Modelo | Tarea | Rama de IA |
|---|--------|-------|-----------|
| 1 | `Salesforce/blip-image-captioning-base` | Image Captioning | Visión |
| 2 | `gpt2` | Text Generation (comentarios) | NLP |
| 3 | `microsoft/speecht5_tts` | Text-to-Speech | Audio |

> Todos se descargan automáticamente en la primera ejecución vía `transformers`.

---

## 📁 Estructura del Proyecto

```
game_commentator/
│
├── app.py                     # Aplicación Gradio — punto de entrada
│
├── modules/
│   ├── __init__.py
│   ├── capture_module.py      # #1 Extracción de frames (OpenCV)
│   ├── vision_module.py       # #2 BLIP captioning + clasificación
│   ├── narration_module.py    # #3 GPT-2 generación de comentarios
│   ├── tts_module.py          # #4 SpeechT5 síntesis de voz
│   └── control_module.py      # #5 Heurística de decisión
│
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

---

## 🚀 Instalación y Ejecución

### 1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⏱️ La primera instalación descarga PyTorch (~2GB) y los modelos de HuggingFace (~2GB adicionales).

### 3. Ejecutar la aplicación

```bash
python app.py
```

La interfaz estará disponible en `http://localhost:7860`

---

## 🧠 Lógica Heurística de Control (Módulo #5)

El sistema implementa las siguientes **reglas de decisión** para decidir cuándo comentar:

| Regla | Condición | Acción |
|-------|-----------|--------|
| Alta prioridad | Escena = "Vida baja" / "Victoria" / "Acción intensa" | Comentar (cooldown reducido) |
| Cooldown mínimo | Tiempo desde último comentario < `MIN_INTERVAL` | No comentar |
| Límite por minuto | > `MAX/MIN_COMMENTS` en los últimos 60s | No comentar |
| Cambio de escena | `scene_label != last_scene` | Comentar |
| Primer frame | `frame_counter == 1` | Comentar |
| Cambio visual | Similitud Jaccard(captions) < umbral | Comentar |
| Repetición | Misma escena, pero han pasado N segundos | Comentar |

La **sensibilidad** (slider en la UI) modula el cooldown y el umbral de Jaccard.

---

## ⚙️ Parámetros Configurables (UI)

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| Analizar cada N segundos | 1–10 | Densidad del muestreo de frames |
| Sensibilidad | 0.1–1.0 | Qué tan fácil es "activar" un comentario |
| Estilo del comentarista | épico / casual / técnico / humorístico | Tono del comentario generado |
| Síntesis de voz | On/Off | Activa TTS (más lento) |

---

## 🎯 Flujo de Datos Detallado

```
Frame PIL (224×224 RGB)
    │
    ├─► BLIP Processor → input_ids + pixel_values
    │       │
    │       └─► BlipForConditionalGeneration.generate()
    │               │
    │               └─► "a screenshot of a video game showing a warrior fighting"
    │
    ├─► VisionModule.classify_scene(caption) → "Combate"
    │       (búsqueda lineal en SCENE_RULES con palabras clave)
    │
    ├─► ControlModule.should_comment("Combate", caption, sensitivity)
    │       (evaluación heurística multi-regla)
    │       → (True, "Cambio de escena: Exploración → Combate")
    │
    ├─► NarrationModule.generate_comment(caption, "Combate", "épico")
    │       │
    │       ├─► prompt = "The warrior clashes in an epic battle! a warrior fighting..."
    │       │
    │       └─► GPT2.generate(prompt, temperature=0.85, top_p=0.92)
    │               │
    │               └─► post_process() → "¡INCREÍBLE! El jugador está en combate intenso."
    │
    └─► TTSModule.synthesize(comment)
            │
            ├─► SpeechT5Processor(text) → input_ids
            │
            └─► SpeechT5ForTextToSpeech.generate_speech(input_ids, speaker_embed, vocoder)
                    └─► numpy array (float32, 16kHz) → archivo .wav
```

---

## 📊 Escenas Detectadas

| Etiqueta | Palabras clave ejemplo | Prioridad |
|----------|------------------------|-----------|
| Combate | fighting, battle, sword, gun, attack | Normal |
| Exploración | walking, exploring, forest, city | Normal |
| Personaje con poca vida | blood, dying, dead, hurt, health low | **Alta** |
| Victoria/Derrota | victory, win, game over, score | **Alta** |
| Cinemática | cutscene, talking, dialogue | Normal |
| Menú/UI | menu, interface, button, inventory | Normal |
| Acción intensa | jumping, racing, chase, speed | **Alta** |
| Descanso/Quietud | standing, idle, peaceful, calm | Normal |
| General | (fallback) | Normal |

---

## ✅ Cumplimiento de Requisitos

| Requisito | Estado | Implementación |
|-----------|--------|----------------|
| #1 Módulo de Captura (.mp4) | ✅ | `capture_module.py` (OpenCV) |
| #2 Módulo de Visión (Opción B HF) | ✅ | BLIP `blip-image-captioning-base` |
| #3 Módulo de Narración NLP (HF) | ✅ | GPT-2 |
| #4 Módulo de Narración TTS (HF) | ✅ | SpeechT5 + HifiGAN |
| #5 Lógica de Control Heurística | ✅ | `control_module.py` (7 reglas) |
| Interfaz (Gradio) | ✅ | `app.py` |
| ≥ 2 modelos Hugging Face | ✅ | BLIP + GPT-2 + SpeechT5 (3 modelos) |
| README con arquitectura | ✅ | Este archivo |

---

## 👥 Notas para el Equipo

- Los modelos se descargan automáticamente en `~/.cache/huggingface/` la primera vez.
- Si la GPU está disponible, PyTorch la usará automáticamente.
- Para videos largos, aumentar "Analizar cada N segundos" reduce el tiempo de procesamiento.
- El TTS puede desactivarse para un análisis más rápido.
