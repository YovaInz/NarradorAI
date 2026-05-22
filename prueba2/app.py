import cv2
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import requests # Para conectar con Ollama
import os

# --- CONFIGURACIÓN DE MODELOS ---
@st.cache_resource
def load_vision_model():
    # Modelo CLIP para Clasificación Zero-Shot
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def load_tts_model():
    # Modelo de Texto a Voz en Español
    return pipeline("text-to-speech", model="facebook/mms-tts-spa")

model_clip, processor_clip = load_vision_model()
tts_pipeline = load_tts_model()

# --- MÓDULO 3: LLM (OLLAMA / GEMMA) ---
def generar_comentario_valorant(evento_detectado):
    """ Envía el evento a Ollama configurado como caster de Valorant """
    prompt = f"""
    Eres un caster/narrador profesional de esports de Valorant, muy energético y entusiasta.
    Reacciona de forma creativa, usando jerga del juego (clutch, spike, headshot, rotación, eco, site, plantar, desactivar), al siguiente suceso:
    Suceso: {evento_detectado}
    Comentario (Corto, máximo 1 frase impactante):
    """
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4", # O el modelo que tengas descargado
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "¡Qué jugada!").strip()
    except:
        return f"¡Atención con eso! Detectamos {evento_detectado} en la partida."

# --- INTERFAZ EN STREAMLIT ---
st.title("🎙️ VCT AI Caster - Narrador Automático de Valorant")
uploaded_file = st.file_uploader("Carga tu clip de Valorant (.mp4)", type=["mp4"])

if uploaded_file is not None:
    # Guardar archivo temporalmente
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
        
    st.video("temp_video.mp4")
    
    # --- PROCESAMIENTO ---
    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Etiquetas que CLIP buscará en el juego
    labels = [
        "captura de pantalla del videojuego Valorant mostrando el menu de compra de armas",
        "captura de pantalla del videojuego Valorant en primera persona disparando un arma, destello de disparo, combate intenso",
        "captura de pantalla del videojuego Valorant en primera persona caminando por el mapa en modo exploracion",
        "captura de pantalla del videojuego Valorant mirando al suelo para plantar o desactivar la spike",
        "captura de pantalla del videojuego Valorant mostrando un indicador rojo de poca vida"
    ]
    
    # Módulo 5: Heurística de Control
    ultimo_evento = ""
    frame_count = 0
    
    st.subheader("📋 Log de la Narración en Vivo")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Decimación: Procesar solo 1 frame por segundo para optimizar recursos
        if frame_count % int(fps) == 0:
            # Convertir frame para CLIP (BGR a RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Módulo 2: Inferencia de Visión
            inputs = processor_clip(text=labels, images=image_rgb, return_tensors="pt", padding=True)
            outputs = model_clip(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).detach().numpy()[0]
            
            # --- NUEVO: Imprimir en consola para depurar ---
            print("\n--- Analizando frame ---")
            for i, label in enumerate(labels):
                print(f"{label}: {probs[i]:.2f}")
            # -----------------------------------------------

            # Obtener la etiqueta con mayor probabilidad
            max_idx = probs.argmax()
            evento_actual = labels[max_idx]
            
            # Lógica Heurística: Solo comentar si el evento cambió significativamente
            if evento_actual != ultimo_evento and probs[max_idx] > 0.70:
                ultimo_evento = evento_actual
                
                # Módulo 3: Narración de Texto
                comentario = generar_comentario_valorant(evento_actual)
                st.write(f"**[Evento]:** {evento_actual.capitalize()}")
                st.info(f"🗣️ *AI Caster:* {comentario}")
                
                # Módulo 4: Narración de Voz (Generar y reproducir temporalmente)
                audio_output = tts_pipeline(comentario)
                st.audio(audio_output["audio"], sample_rate=audio_output["sampling_rate"])
                
        frame_count += 1
        
    cap.release()
    st.success("Análisis del clip terminado.")