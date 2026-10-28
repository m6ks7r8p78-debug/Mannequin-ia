# Création de vidéos de personnages entièrement fictifs avec Wan 2.2 (5B), SANS entraînement.
# 720p vertical (704x1280), 24 images/s, 5 secondes. Tourne sur Kaggle (2 x T4) : la vidéo est
# calculée sur la 1re carte, puis décodée sur la 2e. Une version « fluide » à 48 images/s est aussi créée.
import os, sys, subprocess, time, json, gc, shutil

DEBUT = time.time()
TEST = os.environ.get("MANNEQUIN_TEST") == "1"
SORTIE = os.environ.get("MANNEQUIN_SORTIE", "/kaggle/working")
os.makedirs(os.path.join(SORTIE, "videos"), exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")
DUREE_MAX_HEURES = float(os.environ.get("MANNEQUIN_DUREE_MAX_HEURES", 10.5))

if not TEST and os.environ.get("MANNEQUIN_DEJA_INSTALLE") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "diffusers==0.42.1", "transformers==5.21.0", "accelerate", "ftfy", "sentencepiece"], check=True)

import numpy as np
import torch
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

MODELE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LARGEUR, HAUTEUR, IMAGES, FPS, ETAPES, GUIDAGE = 704, 1280, 121, 24, 40, 5.0
if TEST:
    LARGEUR, HAUTEUR, IMAGES, ETAPES = 64, 128, 9, 2

PERSONNAGES = {
    "balcon_coucher_de_soleil": (
        "Selfie-style vertical video filmed on a smartphone held at arm's length. A woman in her early thirties with a short "
        "black bob haircut and an oversized light denim jacket stands on her apartment balcony at golden hour. She lights a "
        "cigarette with a small lighter, takes a slow drag, then turns her head slightly and exhales a soft cloud of smoke "
        "away from the camera. Relaxed shoulders, calm confident expression, natural subtle movements. Warm sunset light, "
        "city rooftops softly blurred behind her. Photorealistic, natural skin texture, realistic smoke, smooth motion."),
    "voiture_sous_la_pluie": (
        "Selfie-style vertical video filmed by a smartphone fixed on the dashboard of a parked car. A man in his mid thirties "
        "with a short dark beard and a grey hoodie sits in the driver's seat on a rainy evening. He holds a lit cigarette "
        "between two fingers, takes a drag, leans slightly toward the half-open window and exhales the smoke outside. "
        "Thoughtful, calm attitude. Raindrops on the windshield, soft street lights, cinematic realism, smooth natural motion."),
    "cuisine_le_soir": (
        "Selfie-style vertical video filmed on a smartphone standing on a small tripod on a kitchen counter at night. A woman "
        "in her late twenties with long curly red hair and a cream wool sweater leans against the counter, holds a cigarette "
        "between her fingers, takes an unhurried puff and exhales slowly upward, then glances back at the camera with a "
        "relaxed half smile. Warm lamp light, cozy realistic kitchen, photorealistic, realistic smoke, fluid movements."),
    "toit_de_nuit": (
        "Selfie-style vertical video filmed on a smartphone held in one hand on a city rooftop at night. A man in his late "
        "twenties with short blond hair and a black leather jacket shields a lighter from the wind with his hand, lights a "
        "cigarette, takes a deep drag and blows the smoke to the side while the wind carries it away. Cool confident attitude, "
        "neon city lights in the background, photorealistic, natural skin, realistic smoke, smooth handheld motion."),
}
NEGATIF = ("bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
           "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
           "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
           "messy background, three legs, many people in the background, walking backwards, text, watermark")


def journal(*a):
    ligne = " ".join(str(x) for x in a)
    print(ligne, flush=True)
    with open(os.path.join(SORTIE, "journal.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + ligne + "\n")


def heures():
    return (time.time() - DEBUT) / 3600


NB_GPU = torch.cuda.device_count()
if not TEST and NB_GPU == 0:
    raise SystemExit("❌ Pas de carte graphique.")
CALCUL = "cuda:0" if NB_GPU else "cpu"
DECODAGE = "cuda:1" if NB_GPU > 1 else CALCUL
PRECISION = torch.float16 if NB_GPU else torch.float32
if NB_GPU and torch.cuda.get_device_capability(0)[0] >= 8:
    PRECISION = torch.bfloat16
journal(f"⚙️ {NB_GPU} carte(s) · calcul {CALCUL} · décodage {DECODAGE} · {str(PRECISION).replace('torch.', '')}")

# ---------------------------------------------------------------- 1. Textes -> vecteurs (processeur, bf16)
if not TEST:
    from huggingface_hub import snapshot_download
    journal("⬇️ Téléchargement de Wan 2.2 (5B)…")
    DOSSIER = snapshot_download(MODELE, allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                                        "text_encoder/*", "vae/*", "transformer/*"])

textes = list(PERSONNAGES.values()) + [NEGATIF]
if TEST:
    g = torch.Generator().manual_seed(0)
    VECT = {t: torch.randn(min(len(t.split()), 512), 32, generator=g) for t in textes}
else:
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(DOSSIER, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(DOSSIER, subfolder="text_encoder", dtype=torch.bfloat16,
                                           low_cpu_mem_usage=True).eval()
    torch.set_num_threads(os.cpu_count() or 4)
    VECT = {}
    for t in textes:
        e = tok([prompt_clean(t)], max_length=512, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            VECT[t] = enc(e.input_ids, e.attention_mask).last_hidden_state[0].float()
    del enc
    gc.collect()
journal(f"✅ Textes encodés ({heures():.2f} h)")


def completer(v):
    return torch.cat([v, v.new_zeros(512 - v.shape[0], v.shape[1])])[None]


# ---------------------------------------------------------------- 2. Modèles
if TEST:
    torch.manual_seed(0)
    transformer = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                        in_channels=48, out_channels=48, text_dim=32, freq_dim=32, ffn_dim=32,
                                        num_layers=2, rope_max_seq_len=32)
    vae = AutoencoderKLWan(base_dim=12, decoder_base_dim=12, z_dim=48, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True], is_residual=True, in_channels=12, out_channels=12,
                           patch_size=2, scale_factor_spatial=16, scale_factor_temporal=4,
                           latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    sched = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)
else:
    transformer = WanTransformer3DModel.from_pretrained(DOSSIER, subfolder="transformer", torch_dtype=PRECISION)
    vae = AutoencoderKLWan.from_pretrained(DOSSIER, subfolder="vae", torch_dtype=torch.float32)
    sched = UniPCMultistepScheduler.from_pretrained(DOSSIER, subfolder="scheduler")
transformer.to(CALCUL).eval()
vae.to(CALCUL).eval()   # placé avec le calcul pendant la création (le pipeline y crée ses tenseurs)
pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae, scheduler=sched,
                   expand_timesteps=True)
pipe.set_progress_bar_config(disable=True)
if not TEST:
    vae.enable_tiling()
journal(f"✅ Modèles chargés ({heures():.2f} h)")


def decoder(latents):
    """Décodage sur la 2e carte (la 1re garde le gros modèle)."""
    vae.to(DECODAGE)
    z = latents.to(DECODAGE, torch.float32)
    moy = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
    ecart = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
    with torch.no_grad():
        video = vae.decode(z * ecart + moy, return_dict=False)[0]
    vae.to(CALCUL)
    return pipe.video_processor.postprocess_video(video, output_type="np")[0]


FFMPEG = shutil.which("ffmpeg")


def enregistrer(images, chemin, fps):
    imgs = (np.clip(images, 0, 1) * 255).round().astype(np.uint8)
    n, h, w, _ = imgs.shape
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
                    "-movflags", "+faststart", chemin], input=imgs.tobytes(), check=True)


def fluidifier(source, cible):
    """Interpolation de mouvement : 24 -> 48 images/s."""
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", source, "-vf",
                    "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", cible], check=True)


# ---------------------------------------------------------------- 3. Création
neg = completer(VECT[NEGATIF]).to(CALCUL, PRECISION)
for i, (nom, texte) in enumerate(PERSONNAGES.items(), 1):
    if heures() > DUREE_MAX_HEURES:
        journal("⏱️ Temps écoulé, arrêt.")
        break
    t0 = time.time()
    with torch.no_grad():
        latents = pipe(prompt_embeds=completer(VECT[texte]).to(CALCUL, PRECISION), negative_prompt_embeds=neg,
                       height=HAUTEUR, width=LARGEUR, num_frames=IMAGES, num_inference_steps=ETAPES,
                       guidance_scale=GUIDAGE, output_type="latent",
                       generator=torch.Generator(device="cpu").manual_seed(1000 + i)).frames
    if not torch.isfinite(latents).all():
        journal(f"⚠️ {nom} : calcul instable (valeurs infinies), vidéo ignorée.")
        continue
    images = decoder(latents)
    chemin = os.path.join(SORTIE, "videos", f"{i}_{nom}.mp4")
    enregistrer(images, chemin, FPS)
    fluidifier(chemin, chemin.replace(".mp4", "_fluide_48ips.mp4"))
    if NB_GPU:
        torch.cuda.empty_cache()
    journal(f"🎬 {i}/{len(PERSONNAGES)} {nom} : {images.shape[0]} images {images.shape[2]}x{images.shape[1]} "
            f"en {(time.time() - t0) / 60:.1f} min")

journal(f"🏁 Terminé en {heures():.2f} h")
