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


@app.function(gpu="L40S", cpu=8, memory=65536, timeout=24 * 3600,
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


@app.local_entrypoint()
def main(nom_run: str = "essai1", heures: float = 8.0):
    entrainer.remote(nom_run, heures)
