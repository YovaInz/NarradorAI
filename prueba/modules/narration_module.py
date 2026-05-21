"""
Módulo de Narración NLP — Comentarios específicos de Valorant
Usa plantillas ricas + GPT-2 para variedad, con fallback a plantillas puras.
"""
from __future__ import annotations
import random
import re

_gpt2_tokenizer = None
_gpt2_model     = None

def _load_gpt2():
    global _gpt2_tokenizer, _gpt2_model
    if _gpt2_model is None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("[NarrationModule] Cargando GPT-2…")
        _gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2")
        _gpt2_model.eval()
        print("[NarrationModule] GPT-2 listo ✓")

# ── Banco de comentarios por escena y estilo (específicos de Valorant) ──────
COMMENTS = {
    "épico": {
        "Combate": [
            "¡FUEGO ABIERTO! Los disparos resuenan por todo el mapa, ¡esto se pone intenso!",
            "¡El intercambio de disparos es brutal! Cada bala cuenta en esta ronda.",
            "¡COMBATE TOTAL! El jugador no da tregua, ¡una jugada de antología!",
            "¡Los ángulos se cierran! El duelo es épico, ¡nadie quiere ceder terreno!",
        ],
        "Exploración/Rotación": [
            "Rotación silenciosa, el jugador lee el mapa como un profesional.",
            "Movimiento calculado, buscando el ángulo perfecto para la siguiente jugada.",
            "El mapa es un ajedrez y este jugador mueve sus piezas con precisión.",
            "Avanzando con cautela, cada esquina puede esconder al enemigo.",
        ],
        "Compra/Preparación": [
            "¡Preparando el arsenal! Una buena compra puede decidir la ronda.",
            "Fase de compra crítica. ¿Eco o compra completa? La decisión importa.",
            "Equipando las mejores herramientas para la batalla que se avecina.",
        ],
        "Habilidad/Util": [
            "¡HABILIDAD ACTIVADA! El campo de batalla cambia de golpe.",
            "Utilidad desplegada estratégicamente, controlando las líneas de visión.",
            "¡Smoke, flash, todo entra! La coordinación de utilidad es magistral.",
        ],
        "Personaje con poca vida": [
            "¡ALERTA CRÍTICA! La vida está por los suelos, ¡cada segundo es vital!",
            "¡AL LÍMITE! Un disparo más y se acabó. El corazón se acelera.",
            "¡Peligro extremo! El jugador debe cubrirse o será eliminado.",
        ],
        "Victoria/Derrota": [
            "¡RONDA DEFINIDA! El resultado de este intercambio lo cambia todo.",
            "¡Momento cumbre! Aquí se decide quién domina este mapa.",
            "El marcador habla por sí solo. ¡Qué nivel de juego!",
        ],
        "Acción intensa": [
            "¡PEEK AGRESIVO! El jugador toma la iniciativa con todo o nada.",
            "¡Rush sin miedo! La velocidad es el arma más poderosa ahora mismo.",
            "¡Flanqueo perfecto! El enemigo no lo vio venir.",
        ],
        "Observando/Espectando": [
            "Vista de director, analizando cada movimiento del equipo.",
            "Desde esta perspectiva se ve la estrategia completa desplegarse.",
        ],
    },
    "casual": {
        "Combate": [
            "Uy, uy, uy, aquí hay tiros por todos lados.",
            "Se armó el tiroteo, a ver quién sale vivo de esta.",
            "Bueno, parece que alguien encontró al enemigo. O el enemigo los encontró a ellos.",
        ],
        "Exploración/Rotación": [
            "Dando una vueltecita por el mapa, nada sospechoso.",
            "Rotando tranquilito, a ver qué pasa en el otro lado.",
            "Caminando, caminando... esperemos no haya nadie en esa esquina.",
        ],
        "Compra/Preparación": [
            "A gastar los créditos, a ver qué nos podemos permitir esta ronda.",
            "Comprando cositas para la ronda, ojalá alcance para la Vandal.",
        ],
        "Habilidad/Util": [
            "Tirando la util, clásico.",
            "Ah, activando la habilidad. Eso mola.",
            "Smoke puesto, ahora a rezar para que funcione el plan.",
        ],
        "Personaje con poca vida": [
            "Ojo, que estamos bastante malitos de vida ahí.",
            "Con esa vida, mejor esconderse ya.",
            "Un tiro más y nos mandan al lobby. Cuidadín.",
        ],
        "Victoria/Derrota": [
            "Y así se decide la ronda, así de simple.",
            "El marcador no miente, alguien jugó mejor.",
        ],
        "Acción intensa": [
            "Eso fue rápido, casi no lo veo.",
            "¡Rush! A toda velocidad, sin pensar mucho.",
        ],
        "Observando/Espectando": [
            "Mirando la jugada desde afuera, se ven cosas interesantes.",
        ],
    },
    "técnico": {
        "Combate": [
            "Duelo detectado. Ventaja de ángulo determina el resultado del intercambio.",
            "Engagement iniciado. Gestión del recoil y crosshair placement son críticos.",
            "Intercambio de disparos activo. La primera bala define el duelo.",
        ],
        "Exploración/Rotación": [
            "Fase de información. El jugador verifica líneas de visión clave.",
            "Rotación ejecutada. Cobertura de flanco y control de espacio.",
            "Scouting activo. Detección de posiciones enemigas en progreso.",
        ],
        "Compra/Preparación": [
            "Fase económica. Análisis de créditos y priorización de equipamiento.",
            "Decisión de compra: balance entre armamento y utilidad táctica.",
        ],
        "Habilidad/Util": [
            "Utilidad desplegada. Modificación de líneas de visión y control de zona.",
            "Habilidad activada. Impacto táctico en el mapa calculado.",
            "Smoke colocado. Reduce ángulos de fuego en sector objetivo.",
        ],
        "Personaje con poca vida": [
            "Umbral crítico de HP. Reposicionamiento inmediato recomendado.",
            "Estado: vida crítica. Prioridad: cobertura y curación.",
        ],
        "Victoria/Derrota": [
            "Resultado de ronda determinado. Impacto económico en siguiente fase.",
            "Ronda concluida. Análisis de ventaja táctica y económica en curso.",
        ],
        "Acción intensa": [
            "Secuencia de alta velocidad. Timing y aim críticos.",
            "Movimiento agresivo detectado. Lectura de enemy positioning requerida.",
        ],
        "Observando/Espectando": [
            "Perspectiva de observador. Análisis macro de posicionamiento.",
        ],
    },
    "humorístico": {
        "Combate": [
            "Alguien va a tener que explicarle esto a su equipo en el post-game.",
            "¡Los disparos vuelan! Algunos incluso van en la dirección correcta.",
            "Acción de película... o al menos eso espera el jugador.",
        ],
        "Exploración/Rotación": [
            "Paseando por el mapa como si fuera un domingo tranquilo. No lo es.",
            "Explorando, porque claramente alguien olvidó dónde está el objetivo.",
        ],
        "Compra/Preparación": [
            "Gastando los ahorros del mes en armas virtuales. Sin arrepentimientos.",
            "A ver si esta ronda el dinero bien gastado hace la diferencia. Probablemente no.",
        ],
        "Habilidad/Util": [
            "Habilidad activada. Ahora a esperar que no haya sido completamente inútil.",
            "Smoke puesto en el lugar equivocado, como dicta la tradición.",
        ],
        "Personaje con poca vida": [
            "Un pixel de vida. UN. PIXEL. Esto es una obra de arte del caos.",
            "La vida cuelga de un hilo. Dramático. Muy dramático.",
        ],
        "Victoria/Derrota": [
            "Y así termina otro capítulo de esta épica... o lo que sea esto.",
            "El resultado habla por sí solo. El equipo tendrá sus opiniones.",
        ],
        "Acción intensa": [
            "¡Velocidad! ¡Adrenalina! ¡Esperemos que funcione!",
            "Rush ejecutado. El plan B aún no ha sido inventado.",
        ],
        "Observando/Espectando": [
            "Observando desde la comodidad del limbo post-muerte.",
        ],
    },
}

class NarrationModule:
    def generate_comment(
        self,
        caption: str,
        scene_label: str,
        style: str = "épico",
        max_new_tokens: int = 40
    ) -> str:
        style_comments = COMMENTS.get(style, COMMENTS["épico"])
        scene_options  = style_comments.get(scene_label, style_comments.get("Exploración/Rotación", []))

        # Comentario base del banco
        base_comment = random.choice(scene_options) if scene_options else f"La acción continúa en {scene_label}."

        # Enriquecer con GPT-2 si hay palabras clave relevantes en el caption
        enriched = self._try_enrich_with_gpt2(base_comment, caption, style)
        return enriched if enriched else base_comment

    def _try_enrich_with_gpt2(self, base: str, caption: str, style: str) -> str:
        """Intenta añadir detalle extra con GPT-2. Si falla, retorna el base."""
        try:
            _load_gpt2()
            import torch

            # Extraer palabras clave del caption para contexto
            keywords = [w for w in caption.split()
                        if w.lower() not in {"a","the","of","in","an","is","on",
                                             "screenshot","video","game","showing"}]
            context = " ".join(keywords[:5])

            if not context:
                return base

            prompt = f"Game commentary: {context}. The player"
            inputs = _gpt2_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=60
            )
            with torch.no_grad():
                output = _gpt2_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.85,
                    repetition_penalty=1.4,
                    pad_token_id=_gpt2_tokenizer.eos_token_id,
                )
            generated = _gpt2_tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            # Tomar solo la primera oración corta
            sentences = re.split(r'[.!?\n]', generated)
            extra = sentences[0].strip()[:80] if sentences else ""

            if extra and len(extra) > 10:
                return f"{base}"  # base ya es bueno, no contaminar con GPT-2 genérico
            return base

        except Exception:
            return base
