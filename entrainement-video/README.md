# Entraînement IA vidéo — gestes autour de la cigarette, style selfie

Repère automatiquement dans les vidéos les passages avec une cigarette (gestes, posture, attitude,
allumer, tirer, souffler), en privilégiant les plans selfie, puis entraîne un module LoRA pour
l'IA vidéo libre Wan 2.1 (1.3B) et crée des vidéos test.

- `modal_app.py` : version Modal (L40S 48 Go, clips de 5 s, bf16), pilotée par Claude.
- `entrainement_video.ipynb` : version Kaggle (T4 gratuit, clips de 3 s).
- `entrainement_video.py` : le code commun.

Uniquement des vidéos libres de droits, avec des adultes consentants. Pas de contenu à caractère sexuel.
