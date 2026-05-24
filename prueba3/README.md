# 🎙️ Narrador IA — Comentarista Automático de Valorant

Aplicación de escritorio que analiza clips de Valorant fotograma a fotograma, genera comentarios narrativos con un modelo de lenguaje multimodal, y los narra en voz alta en español sincronizados con la reproducción del video.

## Integrantes
* Cynthia Urias Beltran
* Angel Amaury Arredondo González
* Cesar Yovanni Inzunza Aguilar
* Erick David Hermosillo Flores

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│              Punto de entrada — lanza QApplication          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      mainwindow.py                          │
│                   Ventana principal (PyQt6)                 │
│                                                             │
│  ┌─────────────────┐   ┌──────────────┐  ┌──────────────┐   │
│  │   TTSLoader     │   │ QMediaPlayer │  │  CommentBox  │   │
│  │  (QThread)      │   │  (video)     │  │  (QTextEdit) │   │
│  │ Carga el modelo │   └──────┬───────┘  └──────┬───────┘   │
│  │ MMS-TTS en BG   │          │ positionChanged  │ append   │
│  └────────┬────────┘          └──────────────────┘          │
│           │ ready signal                                    │
│  ┌────────▼────────┐                                        │
│  │ TTSAudioThread  │  ← hilo persistente con queue.Queue    │
│  │  (QThread)      │    sin solapamiento de audio           │
│  │ speak(text)     │    velocidad ajustable en tiempo real  │
│  └─────────────────┘                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ start_analysis()
┌───────────────────────────▼─────────────────────────────────┐
│                      analyzer.py                            │
│              VideoAnalyzerSignals (QThread)                 │
│                                                             │
│  1. Abre el video con OpenCV                                │
│  2. Extrae frames cada N segundos                           │
│  3. Los comprime a JPEG 80% / 1280×720                      │
│  4. Los envía al modelo LLM elegido:                        │
│     • Ollama  (local, vía HTTP)                             │
│     • Gemini  (API cloud)                                   │
│     • OpenAI  (API cloud)                                   │
│  5. Filtra respuestas "SKIP"                                │
│  6. Guarda los comentarios en <video>_comments.json         │
│                                                             │
│  Signals: progress(int) · log(str) · finished(str) · error  │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de datos

```
Video (.mp4)
    │
    ▼ OpenCV
Frames JPEG (base64)
    │
    ▼ API (Ollama / Gemini / OpenAI)
Comentarios texto  →  _comments.json
    │
    ▼ QMediaPlayer positionChanged
Comentario visible en pantalla
    │
    ▼ TTSAudioThread (facebook/mms-tts-spa)
Audio WAV en español (velocidad ajustable)
    │
    ▼ sounddevice / winsound
Narración sincronizada
```

---

## Requisitos del sistema

- Python 3.10 o superior
- Windows 10/11, macOS 12+, o Linux con PulseAudio
- GPU opcional pero recomendada para Ollama (VRAM ≥ 8 GB para modelos grandes)
- Conexión a internet solo si usas Gemini u OpenAI

---

## Instalación paso a paso

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Instalar dependencias Python

```bash
pip install -r requirements.txt
```


### 3. Instalar Ollama (solo si usas el modo local)

1. Descarga Ollama desde https://ollama.com/download
2. Instala y ejecuta el servidor:

```bash
ollama serve
```

3. Descarga el modelo multimodal. El proyecto está configurado para `gemma4:e4b` (4B parámetros, ~3 GB):

```bash
ollama run gemma4:e4b
```

> Puedes usar cualquier modelo con capacidad de visión disponible en https://ollama.com/library.
> Para cambiarlo, edita `analyzer.py` → `payload["model"]`.

### 4. Configurar API keys (solo Gemini u OpenAI)

Crea un archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_clave_aqui
OPENAI_API_KEY=tu_clave_aqui
```

Obtén las claves en:
- Gemini: https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

---

## Uso

### Iniciar la aplicación

```bash
python main.py
```

Al arrancar, la app carga automáticamente el modelo TTS (`facebook/mms-tts-spa`) en segundo plano. La barra de estado mostrará "Narración de voz activada ✔" cuando esté listo.

### Flujo de uso

1. **Selecciona un video** — haz clic en "Selecciona un video" y elige un `.mp4`, `.avi` o `.mkv`
2. **Elige el modelo LLM** — selecciona `ollama`, `gemini` u `openai` en el desplegable
3. **Ajusta la tasa de muestra** — define cada cuántos segundos se analiza un frame (por defecto 2s)
4. **Comienza el análisis** — haz clic en "Comenzar Análisis" y espera a que termine
5. **Ajusta la velocidad de voz** — usa el spinner "Vel. voz:" (1.0 = normal, 1.4 = 40% más rápido)
6. **Reproduce el video** — los comentarios aparecen en el panel derecho y se narran en voz alta


### Opciones avanzadas

| Control | Descripción |
|---|---|
| Forzar nuevo análisis | Borra el JSON cacheado y re-analiza el video desde cero |
| Narración de voz | Activa o desactiva el TTS sin reiniciar |
| Vel. voz | Ajusta la velocidad de habla en tiempo real (0.5 – 3.0) |

> Los comentarios se guardan en `<nombre_video>_comments.json` junto al archivo de video. Si ya existe, la app lo reutiliza automáticamente para no gastar créditos de API.

---

## Estructura del proyecto

```
narrador-ia/
├── main.py           # Punto de entrada
├── mainwindow.py     # UI PyQt6, lógica de reproducción y TTS
├── analyzer.py       # Extracción de frames y consulta al LLM
├── requirements.txt  # Dependencias Python
├── .env              # API keys (no incluir en git)
└── README.md
```

---

## Modelos utilizados

| Componente | Modelo | Proveedor |
|---|---|---|
| Visión (local) | `gemma4:e4b` | Ollama |
| Visión (cloud) | `gemini-3.5-flash` | Google |
| Visión (cloud) | `gpt-4o-mini` | OpenAI |
| Texto a voz | `facebook/mms-tts-spa` | HuggingFace |

---