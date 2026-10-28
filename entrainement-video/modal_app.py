"""
Entraînement IA vidéo sur Modal (carte graphique L40S, 48 Go).

Le même code que le carnet Kaggle (entrainement_video.py), lancé sur une carte bien plus puissante :
clips de 5 s, meilleure résolution, calcul en bf16.

Commandes (lancées par Claude depuis son espace de travail) :
    modal volume put mannequin-videos ./videos /            # envoyer les vidéos
    modal run --detach modal_app.py --nom-run essai1        # lancer (continue même si on se déconnecte)
    modal volume get mannequin-resultats essai1 ./resultats # récupérer les résultats

Coût : environ 1,95 $/h sur L40S, couvert par les 30 $ offerts chaque mois par Modal.
La durée est plafonnée (8 h par défaut) pour ne jamais dépasser ce crédit.
"""
import os

import modal

ICI = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("MANNEQUIN_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.15.0", "torchvision==0.29.2",
        "diffusers==0.42.1", "peft==0.21.2", "transformers==5.21.0",
        "accelerate", "open_clip_torch", "ftfy", "sentencepiece", "imageio-ffmpeg",
        "huggingface_hub", "safetensors", "numpy", "pillow",
    )
    .env({"MANNEQUIN_DEJA_INSTALLE": "1", "HF_HOME": "/cache/hf"})
    .add_local_file(os.path.join(ICI, "entrainement_video.py"), "/app/entrainement_video.py")
)

app = modal.App("mannequin-video", image=image)
videos = modal.Volume.from_name("mannequin-videos", create_if_missing=True)
resultats = modal.Volume.from_name("mannequin-resultats", create_if_missing=True)
cache = modal.Volume.from_name("mannequin-cache", create_if_missing=True)   # modèles téléchargés une seule fois


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=24 * 3600,
              volumes={"/videos": videos, "/resultats": resultats, "/cache": cache})
def entrainer(nom_run: str, duree_max_heures: float = 8.0):
    import runpy
    import threading

    os.environ.update(
        MANNEQUIN_ENTREES="/videos",
        MANNEQUIN_SORTIE=f"/resultats/{nom_run}",
        MANNEQUIN_DUREE_MAX_HEURES=str(duree_max_heures),
    )
    os.makedirs(f"/resultats/{nom_run}", exist_ok=True)

    # Sauvegarde régulière des résultats (journal, modules intermédiaires) pendant l'entraînement
    fini = threading.Event()

    def sauvegarde_reguliere():
        while not fini.wait(600):
            try:
                resultats.commit()
            except Exception as e:
                print("sauvegarde intermédiaire impossible :", e)

    threading.Thread(target=sauvegarde_reguliere, daemon=True).start()
    try:
        runpy.run_path("/app/entrainement_video.py", run_name="__main__")
    finally:
        fini.set()
        resultats.commit()
        cache.commit()


@app.function(cpu=4, memory=16384, timeout=3 * 3600, volumes={"/cache": cache})
def precharger():
    """Télécharge une fois pour toutes les modèles dans le cache (sans carte graphique, presque gratuit)."""
    from huggingface_hub import snapshot_download
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "vae/*", "transformer/*"])
    snapshot_download("Salesforce/blip-image-captioning-large")
    import open_clip
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    cache.commit()
    return "modèles en cache"


@app.function(gpu=GPU, timeout=600, volumes={"/cache": cache})
def verifier():
    """Vérifie en 1 minute que la carte graphique et les bibliothèques fonctionnent."""
    import torch, diffusers, peft, transformers
    from diffusers import WanTransformer3DModel
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    ok = torch.isfinite(x @ x).all().item()
    t = WanTransformer3DModel(num_layers=1, num_attention_heads=2, attention_head_dim=64, text_dim=64,
                              freq_dim=64, ffn_dim=128).to("cuda", torch.bfloat16)
    y = t(torch.randn(1, 16, 3, 16, 16, device="cuda", dtype=torch.bfloat16),
          torch.tensor([500.0], device="cuda"), torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16),
          return_dict=False)[0]
    return {"carte": torch.cuda.get_device_name(0),
            "memoire_go": round(torch.cuda.get_device_properties(0).total_memory / 2**30),
            "bf16": torch.cuda.is_bf16_supported(), "calcul_ok": ok and bool(torch.isfinite(y).all()),
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "peft": peft.__version__, "transformers": transformers.__version__}


@app.local_entrypoint()
def main(nom_run: str = "essai1", heures: float = 8.0):
    entrainer.remote(nom_run, heures)
