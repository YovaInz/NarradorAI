"""
Módulo de Visión — Reglas optimizadas para Valorant
"""
from __future__ import annotations
from PIL import Image

_blip_processor = None
_blip_model     = None

def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        model_id = "Salesforce/blip-image-captioning-base"
        print(f"[VisionModule] Cargando {model_id}…")
        _blip_processor = BlipProcessor.from_pretrained(model_id)
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        _blip_model.eval()
        print("[VisionModule] BLIP listo ✓")

# Reglas ajustadas para Valorant — orden de prioridad
SCENE_RULES = [
    ("Personaje con poca vida", [
        "blood", "dying", "dead", "hurt", "wounded", "injury", "red screen"
    ]),
    ("Compra/Preparación", [
        "shop", "buy", "store", "purchase", "menu", "interface", "hud",
        "icon", "button", "settings", "inventory", "panel"
    ]),
    ("Combate", [
        "shooting", "fighting", "firing", "gun", "rifle", "pistol",
        "sniper", "weapon", "attack", "explosion", "shoot", "aiming",
        "crosshair", "kill", "hit", "bullet"
    ]),
    ("Victoria/Derrota", [
        "victory", "defeat", "win", "lose", "game over", "score board",
        "scoreboard", "end", "round over", "spike", "defuse", "plant"
    ]),
    ("Exploración/Rotación", [
        "walking", "running", "moving", "hallway", "corridor", "tunnel",
        "street", "alley", "wall", "building", "outdoor", "map", "area"
    ]),
    ("Habilidad/Util", [
        "smoke", "flash", "grenade", "ability", "skill", "sphere",
        "orb", "effect", "light", "particle", "robot", "drone",
        "turret", "barrier", "wall ability"
    ]),
    ("Observando/Espectando", [
        "watching", "spectating", "observer", "camera", "replay",
        "cinematic", "cutscene"
    ]),
    ("Acción intensa", [
        "jumping", "climbing", "rushing", "pushing", "rotating",
        "flanking", "peeking", "peek"
    ]),
]

class VisionModule:
    def caption_frame(self, image: Image.Image) -> str:
        _load_blip()
        import torch
        conditional_text = "a screenshot of a first person shooter game showing"
        inputs = _blip_processor(
            images=image,
            text=conditional_text,
            return_tensors="pt"
        )
        with torch.no_grad():
            output_ids = _blip_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True
            )
        caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def classify_scene(self, caption: str) -> str:
        caption_lower = caption.lower()
        for label, keywords in SCENE_RULES:
            if any(kw in caption_lower for kw in keywords):
                return label
        return "Exploración/Rotación"  # fallback más común en Valorant

    def get_scene_keywords(self, scene_label: str) -> list:
        for label, keywords in SCENE_RULES:
            if label == scene_label:
                return keywords
        return []
