# %% [markdown]
# # 🎬 Entraînement IA vidéo — gestes autour de la cigarette, style selfie
#
# Ce carnet fait **tout automatiquement**, du début à la fin :
#
# 1. il regarde toutes tes vidéos et repère seul les passages avec une cigarette
#    (gestes, posture, attitude, allumer, tirer, souffler), en privilégiant les vidéos selfie ;
# 2. il les découpe en petits clips et les décrit avec des mots (dont le cadrage selfie) ;
# 3. il entraîne un module (LoRA) pour l'IA vidéo libre **Wan 2.1** (1,3 milliard de paramètres) ;
# 4. il crée des vidéos test avec ce que l'IA a appris.
#
# **Avant de lancer** (réglages à droite du carnet, menu « Settings ») :
# - **Accelerator** : `GPU T4 x2` ;
# - **Internet** : activé ;
# - tes vidéos ajoutées avec **Add Input** (bouton à droite).
#
# Puis **Save Version → Save & Run All (Commit)**. Tu peux fermer la page : tout tourne seul (3 à 6 h).
# (Le même code tourne aussi sur Modal, avec une carte graphique bien plus puissante : voir `modal_app.py`.)
# Les résultats seront dans l'onglet **Output** de la version.
#
# Uniquement des vidéos libres de droits, avec des adultes consentants.

# %%
# ======================================================================
#  RÉGLAGES — tu peux tout laisser tel quel.
# ======================================================================
MOT_CLE = "smkmotion"  # mot magique qui active le style appris
DESCRIPTION_STYLE = ("a person smoking a cigarette with natural gestures, relaxed posture and confident attitude, "
                     "inhaling and exhaling smoke")
PHRASE_SELFIE = ("selfie-style video, the person films themselves with a handheld phone or a phone on a tripod, "
                 "looking into the camera")

# Préparation des vidéos
NB_CLIPS_MAX = 240          # nombre maximum de clips utilisés pour apprendre
CLIPS_MAX_PAR_VIDEO = 10    # plusieurs moments par vidéo, sans trop favoriser une seule personne
SECONDES_ANALYSE_MAX = 900  # on analyse au plus les 15 premières minutes de chaque vidéo
SEUIL_FUMER = 0.20          # 0 à 1 : à quel point l'IA doit être sûre qu'une cigarette est présente
BONUS_SELFIE = 0.5          # priorité donnée aux plans selfie (0 = aucune)

# Entraînement
ETAPES = 1600
TAUX_APPRENTISSAGE = 1e-4
RANG_LORA = 32
SAUVEGARDE_TOUTES = 400
DUREE_MAX_HEURES = 10.5     # Kaggle coupe à 12 h : on s'arrête avant pour garder les résultats

# Vidéos test créées à la fin
FORMAT_TEST = "portrait"    # "portrait" (TikTok) ou "paysage"
IMAGES_TEST = 49            # 49 images = 3 secondes, 81 = 5 secondes (plus long à créer)
ETAPES_TEST = 30
PROMPTS_TEST = [
    "selfie-style video, a woman with long dark hair films herself with her phone on a tripod in her living room, "
    "slowly takes a drag of her cigarette and exhales a cloud of smoke, relaxed posture, confident attitude, soft daylight",
    "selfie-style video, a man in a black leather jacket holds his phone at arm's length on a balcony at night, "
    "lights a cigarette, inhales deeply and blows smoke upwards, city lights behind him",
    "selfie-style video, a woman with short blond hair sitting in her car films herself, holds a cigarette between her fingers, "
    "takes a puff and exhales slowly through the open window, natural light",
]
NEGATIF = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)

# %%
# ======================================================================
#  Installation et vérifications (ne rien changer en dessous)
# ======================================================================
import os, sys, subprocess, time, json, math, random, gc, glob, shutil

DEBUT = time.time()
TEST = os.environ.get("MANNEQUIN_TEST") == "1"  # mode simulation (utilisé pour vérifier le carnet)
DUREE_MAX_HEURES = float(os.environ.get("MANNEQUIN_DUREE_MAX_HEURES", DUREE_MAX_HEURES))
ESSAI_RAPIDE = os.environ.get("MANNEQUIN_ESSAI_RAPIDE") == "1"   # essai technique de quelques minutes
if ESSAI_RAPIDE:
    NB_CLIPS_MAX, ETAPES, SAUVEGARDE_TOUTES, SEUIL_FUMER = 6, 30, 15, 0.0
    PROMPTS_TEST, IMAGES_TEST, ETAPES_TEST = PROMPTS_TEST[:1], 17, 10
if TEST:
    NB_CLIPS_MAX, ETAPES, SAUVEGARDE_TOUTES, IMAGES_TEST, ETAPES_TEST = 12, 8, 4, 9, 2
    ETAPES = int(os.environ.get("MANNEQUIN_TEST_ETAPES", ETAPES))

if not TEST and os.environ.get("MANNEQUIN_DEJA_INSTALLE") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "diffusers==0.42.1", "peft==0.21.2", "transformers==5.21.0",
                    "accelerate", "open_clip_torch", "ftfy", "sentencepiece", "imageio-ffmpeg"], check=True)

import numpy as np
import torch
from PIL import Image

SORTIE = os.environ.get("MANNEQUIN_SORTIE", "/kaggle/working")
ENTREES = os.environ.get("MANNEQUIN_ENTREES", "/kaggle/input")
os.makedirs(SORTIE, exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")
MODELE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and not TEST:
    raise SystemExit("❌ Pas de carte graphique. Dans Settings → Accelerator, choisis « GPU T4 x2 », puis relance.")
if DEVICE == "cuda":
    for i in range(torch.cuda.device_count()):
        print("✅ Carte graphique :", torch.cuda.get_device_name(i))
torch.set_num_threads(os.cpu_count() or 4)

# Grosse carte graphique (Modal : L40S 48 Go, A100…) => clips plus longs et plus nets
MEMOIRE_GO = torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE == "cuda" else 0
GROS_GPU = MEMOIRE_GO >= 30 or (TEST and os.environ.get("MANNEQUIN_TEST_GROS") == "1")
# Précision de calcul : bf16 si la carte le permet (plus stable), sinon fp16 (T4)
_choix = os.environ.get("MANNEQUIN_TEST_PRECISION") if TEST else None
if _choix:
    PRECISION = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[_choix]
elif DEVICE == "cuda":
    PRECISION = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
else:
    PRECISION = torch.float32
AMP = PRECISION != torch.float32
if GROS_GPU and not ESSAI_RAPIDE:
    NB_CLIPS_MAX = max(NB_CLIPS_MAX, 300)
    ETAPES = int(os.environ.get("MANNEQUIN_ETAPES", 2400))
    IMAGES_TEST, ETAPES_TEST = 81, 40
    if TEST:
        NB_CLIPS_MAX, ETAPES, IMAGES_TEST, ETAPES_TEST = 12, 8, 17, 2
print(f"⚙️ Mémoire carte : {MEMOIRE_GO:.0f} Go · précision : {str(PRECISION).replace('torch.', '')} · "
      f"{'grande' if GROS_GPU else 'petite'} configuration")

try:
    import imageio_ffmpeg
    FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
print("ffmpeg :", FFMPEG)

def heures():
    return (time.time() - DEBUT) / 3600

def journal(*a):
    ligne = " ".join(str(x) for x in a)
    print(ligne, flush=True)
    with open(os.path.join(SORTIE, "journal.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + ligne + "\n")

def liberer():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
# ======================================================================
#  1. Trouver les vidéos
# ======================================================================
EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp")
VIDEOS = sorted(p for p in glob.glob(os.path.join(ENTREES, "**", "*"), recursive=True)
                if p.lower().endswith(EXTENSIONS))
journal(f"🎞️ {len(VIDEOS)} vidéo(s) trouvée(s)")
if not VIDEOS:
    raise SystemExit("❌ Aucune vidéo trouvée. Ajoute ton dataset de vidéos avec « Add Input », puis relance.")

IMAGES_CLIP = 81 if GROS_GPU else 49   # 81 images ≈ 5 s, 49 ≈ 3 s, à 16 images/s (format attendu par Wan)
FPS_CLIP = 16
DUREE_CLIP = IMAGES_CLIP / FPS_CLIP
FPS_ANALYSE = 2

def infos_video(chemin):
    """Durée et orientation (en tenant compte de la rotation des vidéos d'iPhone)."""
    if FFPROBE:
        cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height:stream_side_data=rotation:stream_tags=rotate:format=duration",
               "-of", "json", chemin]
        try:
            d = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout or "{}")
            s = (d.get("streams") or [{}])[0]
            w, h = int(s.get("width", 0)), int(s.get("height", 0))
            rot = 0
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(float(sd["rotation"]))
            rot = rot or int(float((s.get("tags") or {}).get("rotate", 0) or 0))
            if abs(rot) % 180 == 90:
                w, h = h, w
            duree = float((d.get("format") or {}).get("duration", 0) or 0)
            if w and h and duree:
                return duree, w, h
        except Exception:
            pass
    # Secours : on décode pour mesurer
    sortie = subprocess.run([FFMPEG, "-hide_banner", "-i", chemin, "-f", "null", "-"],
                            capture_output=True, text=True).stderr
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", sortie)
    duree = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0
    m = re.search(r", (\d{2,5})x(\d{2,5})", sortie)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (16, 9)
    return duree, w, h

def lire_images(chemin, debut, duree, fps, largeur, hauteur, nb_max=None):
    """Décode une portion de vidéo en images RVB recadrées (largeur x hauteur)."""
    vf = (f"fps={fps},scale={largeur}:{hauteur}:force_original_aspect_ratio=increase,"
          f"crop={largeur}:{hauteur},setsar=1")
    cmd = [FFMPEG, "-v", "error", "-ss", f"{debut:.3f}", "-i", chemin, "-t", f"{duree:.3f}",
           "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    if nb_max:
        cmd += ["-frames:v", str(nb_max)]
    cmd += ["-"]
    brut = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    taille = largeur * hauteur * 3
    n = len(brut) // taille
    return np.frombuffer(brut[: n * taille], dtype=np.uint8).reshape(n, hauteur, largeur, 3).copy()

# %%
# ======================================================================
#  2. Repérer automatiquement les passages où la personne fume (CLIP)
# ======================================================================
POSITIFS = ["a photo of a person smoking a cigarette",
            "a person exhaling cigarette smoke",
            "a close-up of a person holding a cigarette to their lips",
            "a person lighting a cigarette",
            "a person holding a lit cigarette between their fingers",
            "a selfie of a person with a cigarette"]
NEGATIFS = ["a photo of a person", "a photo of a room", "a photo of text on a screen",
            "a blurry dark photo", "a photo of an object on a table", "a person eating food"]
SELFIE_OUI = ["a selfie video of a person filming themselves with a phone",
              "a person holding a phone at arm's length, filming themselves",
              "a person filming themselves with a phone on a tripod, looking at the camera"]
SELFIE_NON = ["a video of a person filmed by someone else from a distance",
              "a wide shot of a person who is not looking at the camera",
              "a photo with no person in it"]

def charger_clip():
    if TEST:
        def faux(images):
            m = images.reshape(len(images), -1).mean(1) / 255.0
            return np.clip(m * 1.6, 0, 1), np.clip(1.2 - m, 0, 1)
        return faux
    import open_clip
    modele, _, pretraitement = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    modele = modele.to(DEVICE).eval()
    def textes(liste):
        with torch.no_grad():
            t = modele.encode_text(tok(liste).to(DEVICE)).float()
        return t / t.norm(dim=-1, keepdim=True)
    t_cig, t_selfie = textes(POSITIFS + NEGATIFS), textes(SELFIE_OUI + SELFIE_NON)

    def scorer(images):
        cig, selfie = [], []
        for i in range(0, len(images), 64):
            lot = torch.stack([pretraitement(Image.fromarray(x)) for x in images[i:i + 64]]).to(DEVICE)
            with torch.no_grad():
                v = modele.encode_image(lot).float()
                v = v / v.norm(dim=-1, keepdim=True)
                p1 = (100.0 * v @ t_cig.T).softmax(dim=-1)
                p2 = (100.0 * v @ t_selfie.T).softmax(dim=-1)
            cig.append(p1[:, :len(POSITIFS)].sum(-1).cpu().numpy())
            selfie.append(p2[:, :len(SELFIE_OUI)].sum(-1).cpu().numpy())
        if not cig:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(cig), np.concatenate(selfie)
    return scorer

def coupures(images):
    """Repère les changements de plan (pour ne pas couper un clip au milieu d'un changement)."""
    if len(images) < 2:
        return np.zeros(len(images), dtype=bool)
    petites = images[:, ::7, ::7].astype(np.float32).mean(-1)
    diff = np.abs(petites[1:] - petites[:-1]).mean((1, 2))
    return np.concatenate([[False], diff > 38])

scorer = charger_clip()
candidats = {}   # vidéo -> liste (priorité, début, orientation, score cigarette, selfie ?)
ignorees = 0
for n, chemin in enumerate(VIDEOS, 1):
    try:
        duree, w, h = infos_video(chemin)
        if duree < DUREE_CLIP + 0.2:
            ignorees += 1
            continue
        imgs = lire_images(chemin, 0, min(duree, SECONDES_ANALYSE_MAX), FPS_ANALYSE, 224, 224)
        if len(imgs) < 3:
            ignorees += 1
            continue
        s, sf = scorer(imgs)
        cut = coupures(imgs)
        fen = int(math.ceil(DUREE_CLIP * FPS_ANALYSE))
        options = []
        for i in range(0, len(imgs) - fen + 1):
            if cut[i + 1:i + fen].any():
                continue
            cig, selfie = float(s[i:i + fen].mean()), float(sf[i:i + fen].mean())
            options.append((cig + BONUS_SELFIE * selfie, i / FPS_ANALYSE, cig, selfie >= 0.5))
        options.sort(reverse=True)
        choisis = []
        orient = "portrait" if h > w * 1.05 else "paysage"
        for prio, deb, cig, est_selfie in options:   # moments variés qui ne se chevauchent pas
            if all(abs(deb - c[1]) >= DUREE_CLIP for c in choisis):
                choisis.append((prio, deb, orient, cig, est_selfie))
            if len(choisis) >= CLIPS_MAX_PAR_VIDEO:
                break
        candidats[chemin] = choisis
        if n % 10 == 0 or n == len(VIDEOS):
            journal(f"   analyse {n}/{len(VIDEOS)} vidéos…")
    except Exception as e:
        ignorees += 1
        journal(f"   ⚠️ vidéo illisible ignorée : {os.path.basename(chemin)} ({e})")

del scorer
liberer()

def selection(seuil):
    """Tour par tour entre les vidéos pour varier les personnes."""
    pools = {k: [c for c in v if c[3] >= seuil] for k, v in candidats.items()}
    res, tour = [], 0
    while len(res) < NB_CLIPS_MAX and any(len(p) > tour for p in pools.values()):
        lot = sorted(((p[tour], k) for k, p in pools.items() if len(p) > tour), reverse=True)
        for c, k in lot:
            if len(res) < NB_CLIPS_MAX:
                res.append((k, c[1], c[2], c[3], c[4]))   # vidéo, début, orientation, score, selfie
        tour += 1
    return res

seuil = SEUIL_FUMER
CLIPS = selection(seuil)
while len(CLIPS) < min(24, sum(len(v) for v in candidats.values())) and seuil > 0.05:
    seuil = round(seuil - 0.05, 2)
    CLIPS = selection(seuil)
journal(f"✂️ {len(CLIPS)} passages retenus (seuil {seuil}), dont {sum(c[4] for c in CLIPS)} en selfie ; "
        f"{ignorees} vidéo(s) ignorée(s) (trop courtes ou illisibles)")
if len(CLIPS) < (2 if ESSAI_RAPIDE else 8):
    raise SystemExit(f"❌ Trop peu de passages utilisables. Ajoute des vidéos (d'au moins {DUREE_CLIP + 1:.0f} secondes) où l'on voit bien la cigarette.")

# %%
# ======================================================================
#  3. Découper les clips et les décrire avec des mots (BLIP)
# ======================================================================
TAILLES = ({"paysage": (624, 352), "portrait": (352, 624)} if GROS_GPU
           else {"paysage": (512, 288), "portrait": (288, 512)})   # (largeur, hauteur) pour l'apprentissage
if TEST:
    TAILLES = {"paysage": (64, 32), "portrait": (32, 64)}

def charger_legendeur():
    if TEST:
        return lambda img: "a person sitting indoors"
    from transformers import BlipProcessor, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    mod = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                       dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)

    def legender(img):
        entree = proc(Image.fromarray(img), "a photograph of", return_tensors="pt").to(DEVICE, mod.dtype)
        with torch.no_grad():
            sortie = mod.generate(**entree, max_new_tokens=30)
        return proc.decode(sortie[0], skip_special_tokens=True).strip()
    return legender

legender = charger_legendeur()
DONNEES = []   # (images uint8 [N,H,W,3], légende, infos)
for k, (chemin, debut, orient, sc, est_selfie) in enumerate(CLIPS, 1):
    W, H = TAILLES[orient]
    imgs = lire_images(chemin, debut, DUREE_CLIP + 0.5, FPS_CLIP, W, H, nb_max=IMAGES_CLIP)
    if len(imgs) < IMAGES_CLIP:
        continue
    texte = legender(imgs[IMAGES_CLIP // 2])
    selfie_txt = f"{PHRASE_SELFIE}, " if est_selfie else ""
    legende = f"{MOT_CLE}, {selfie_txt}{texte}, {DESCRIPTION_STYLE}"
    DONNEES.append((imgs, legende, f"{os.path.basename(chemin)} @ {debut:.1f}s (score {sc:.2f})"))
    if k % 20 == 0:
        journal(f"   clips préparés : {k}/{len(CLIPS)}")
del legender
liberer()
journal(f"✅ {len(DONNEES)} clips prêts. Exemple de légende : « {DONNEES[0][1]} »")

# Planche d'aperçu des passages choisis (pour vérifier d'un coup d'œil)
vignettes = [Image.fromarray(d[0][IMAGES_CLIP // 2]).resize((160, 160)) for d in DONNEES[:30]]
planche = Image.new("RGB", (160 * 6, 160 * math.ceil(len(vignettes) / 6)), "black")
for i, v in enumerate(vignettes):
    planche.paste(v, ((i % 6) * 160, (i // 6) * 160))
planche.save(os.path.join(SORTIE, "apercu_passages_choisis.jpg"), quality=85)
with open(os.path.join(SORTIE, "legendes.txt"), "w", encoding="utf-8") as f:
    for _, leg, info in DONNEES:
        f.write(f"{info} : {leg}\n")

# %%
# ======================================================================
#  4. Transformer les textes en « vecteurs » (encodeur UMT5, sur le processeur)
# ======================================================================
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

if not TEST:
    from huggingface_hub import snapshot_download
    journal("⬇️ Téléchargement de Wan 2.1 (1.3B)… (peut prendre 10-20 min)")
    DOSSIER_MODELE = snapshot_download(MODELE, allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "vae/*", "transformer/*"])
    journal("   espace disque restant :", shutil.disk_usage("/tmp").free // 2**30, "Go")

MAX_TOKENS = 512

def encoder_textes(textes):
    """Même calcul que Wan, mais sur le processeur en bf16 (évite les erreurs du fp16 sur T4)."""
    if TEST:
        g = torch.Generator().manual_seed(0)
        return {t: torch.randn(min(len(t.split()) + 1, MAX_TOKENS), 32, generator=g) for t in textes}
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(DOSSIER_MODELE, subfolder="tokenizer")
    # Grosse carte : encodage rapide sur la carte (bf16). Sinon : sur le processeur (bf16, évite le fp16 du T4).
    lieu = DEVICE if (GROS_GPU and PRECISION == torch.bfloat16) else "cpu"
    enc = UMT5EncoderModel.from_pretrained(DOSSIER_MODELE, subfolder="text_encoder",
                                           dtype=torch.bfloat16, low_cpu_mem_usage=True).to(lieu).eval()
    res = {}
    for i, t in enumerate(textes, 1):
        e = tok([prompt_clean(t)], max_length=MAX_TOKENS, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(e.input_ids.to(lieu), e.attention_mask.to(lieu)).last_hidden_state[0]
        res[t] = h.float().cpu()
        if i % 25 == 0:
            journal(f"   textes encodés : {i}/{len(textes)}")
    del enc
    liberer()
    return res

def completer(v):
    """Complète à 512 positions avec des zéros, comme Wan."""
    return torch.cat([v, v.new_zeros(MAX_TOKENS - v.shape[0], v.shape[1])])[None]

PROMPTS_COMPLETS = [f"{MOT_CLE}, {p}" for p in PROMPTS_TEST]
tous_textes = list(dict.fromkeys([d[1] for d in DONNEES] + [""] + PROMPTS_COMPLETS + [NEGATIF]))
TEXTES = encoder_textes(tous_textes)
journal(f"✅ {len(TEXTES)} textes encodés ({heures():.2f} h écoulées)")

# %%
# ======================================================================
#  5. Compresser les clips en « latents » (VAE de Wan)
# ======================================================================
from diffusers import AutoencoderKLWan, WanTransformer3DModel

def charger_vae():
    if TEST:
        torch.manual_seed(0)
        return AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                temperal_downsample=[False, True, True])
    return AutoencoderKLWan.from_pretrained(DOSSIER_MODELE, subfolder="vae", torch_dtype=torch.float32)

vae = charger_vae().to(DEVICE).eval()
MOY = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
ECART = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

def vers_latent(imgs):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
    with torch.no_grad():
        z = vae.encode(x.to(DEVICE)).latent_dist.sample().cpu()
    return ((z - MOY) / ECART).half()

LATENTS = []   # (latent, légende)
for i, (imgs, leg, _) in enumerate(DONNEES, 1):
    LATENTS.append((vers_latent(imgs), leg))
    LATENTS.append((vers_latent(imgs[:, :, ::-1]), leg))   # version miroir : double les exemples
    if i % 20 == 0:
        journal(f"   clips compressés : {i}/{len(DONNEES)}")
del DONNEES
liberer()
journal(f"✅ {len(LATENTS)} exemples d'apprentissage, taille {tuple(LATENTS[0][0].shape)}")

# %%
# ======================================================================
#  6. Entraînement du module LoRA
# ======================================================================
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers import WanPipeline

def charger_transformer():
    if TEST:   # mini-modèle aléatoire, chargé comme le vrai (fp16 + modules sensibles en fp32)
        import tempfile
        torch.manual_seed(0)
        d = tempfile.mkdtemp()
        WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                              in_channels=16, out_channels=16, text_dim=32, freq_dim=32, ffn_dim=32,
                              num_layers=2, rope_max_seq_len=32).save_pretrained(d)
        return WanTransformer3DModel.from_pretrained(d, torch_dtype=PRECISION)
    return WanTransformer3DModel.from_pretrained(DOSSIER_MODELE, subfolder="transformer", torch_dtype=PRECISION)

transformer = charger_transformer().to(DEVICE)
transformer.requires_grad_(False)
transformer.enable_gradient_checkpointing()
transformer.add_adapter(LoraConfig(
    r=RANG_LORA, lora_alpha=RANG_LORA, init_lora_weights=True,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]))
params = [p for p in transformer.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()   # le module appris reste en haute précision
journal(f"🧠 Module LoRA : {sum(p.numel() for p in params) / 1e6:.1f} M paramètres à apprendre")

optim = torch.optim.AdamW(params, lr=TAUX_APPRENTISSAGE, weight_decay=1e-4)
scaler = torch.amp.GradScaler(DEVICE, enabled=PRECISION == torch.float16)   # utile seulement en fp16
DECALAGE = 3.0   # décalage du bruit (réglage standard de Wan)
VIDE = completer(TEXTES[""])

def sauvegarder(nom):
    dossier = os.path.join(SORTIE, nom)
    WanPipeline.save_lora_weights(dossier, transformer_lora_layers=get_peft_model_state_dict(transformer))
    journal(f"💾 Module sauvegardé : {nom}/pytorch_lora_weights.safetensors")

transformer.train()
ordre, pertes, nan_suite, t0 = [], [], 0, time.time()
etape = 0
for etape in range(1, ETAPES + 1):
    if not ordre:
        ordre = list(range(len(LATENTS)))
        random.shuffle(ordre)
    z0, leg = LATENTS[ordre.pop()]
    z0 = z0.to(DEVICE, torch.float32)
    texte = VIDE if random.random() < 0.1 else completer(TEXTES[leg])   # 10 % sans texte (meilleur guidage)
    texte = texte.to(DEVICE)
    u = torch.sigmoid(torch.randn(1))
    sigma = (DECALAGE * u / (1 + (DECALAGE - 1) * u)).item()
    bruit = torch.randn_like(z0)
    zt = (1 - sigma) * z0 + sigma * bruit
    cible = bruit - z0
    for g in optim.param_groups:
        g["lr"] = TAUX_APPRENTISSAGE * min(1.0, etape / 50)
    with torch.autocast(DEVICE, dtype=PRECISION, enabled=AMP):
        pred = transformer(hidden_states=zt.to(PRECISION),
                           timestep=torch.tensor([sigma * 1000.0], device=DEVICE),
                           encoder_hidden_states=texte.to(PRECISION),
                           return_dict=False)[0]
    perte = torch.nn.functional.mse_loss(pred.float(), cible)
    if not torch.isfinite(perte):
        nan_suite += 1
        optim.zero_grad(set_to_none=True)
        if nan_suite >= 25:
            journal("❌ Calculs instables (valeurs infinies). Baisse TAUX_APPRENTISSAGE à 5e-5 et relance.")
            break
        continue
    nan_suite = 0
    scaler.scale(perte).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    pertes.append(perte.item())
    if etape % 25 == 0 or etape == 1:
        vitesse = (time.time() - t0) / etape
        reste = vitesse * (ETAPES - etape) / 3600
        journal(f"   étape {etape}/{ETAPES} · erreur {np.mean(pertes[-25:]):.4f} · {vitesse:.1f} s/étape · reste ≈ {reste:.1f} h")
    if etape % SAUVEGARDE_TOUTES == 0 and etape < ETAPES:
        sauvegarder(f"lora_etape_{etape}")
    if heures() > DUREE_MAX_HEURES - 0.75:
        journal("⏱️ Temps presque écoulé : arrêt anticipé pour garder les résultats.")
        break

sauvegarder("lora_final")
with open(os.path.join(SORTIE, "courbe_erreur.json"), "w") as f:
    json.dump(pertes, f)
del optim, LATENTS
liberer()
journal(f"✅ Entraînement terminé ({etape} étapes, {heures():.2f} h écoulées)")

# %%
# ======================================================================
#  7. Créer des vidéos test avec le module appris
# ======================================================================
from diffusers import UniPCMultistepScheduler

def exporter_video(images, chemin, fps=16):
    """Enregistre les images (0..1) en MP4 lisible sur iPhone."""
    imgs = (np.clip(np.asarray(images), 0, 1) * 255).round().astype(np.uint8)
    n, h, w, _ = imgs.shape
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", chemin]
    subprocess.run(cmd, input=imgs.tobytes(), check=True)

transformer.eval()
if AMP:   # le module appris passe dans la précision du modèle, le reste ne bouge pas
    for p in params:
        p.data = p.data.to(PRECISION)
if TEST:
    planificateur = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True,
                                            flow_shift=3.0, num_train_timesteps=1000)
else:
    planificateur = UniPCMultistepScheduler.from_pretrained(DOSSIER_MODELE, subfolder="scheduler", flow_shift=3.0)
pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae, scheduler=planificateur)
pipe.set_progress_bar_config(disable=True)
if not TEST:
    vae.enable_tiling()

LARG, HAUT = (480, 832) if FORMAT_TEST == "portrait" else (832, 480)
if TEST:
    LARG, HAUT = (32, 64) if FORMAT_TEST == "portrait" else (64, 32)
dossier_videos = os.path.join(SORTIE, "videos_test")
os.makedirs(dossier_videos, exist_ok=True)
neg = completer(TEXTES[NEGATIF]).to(DEVICE, transformer.dtype)
for i, texte in enumerate(PROMPTS_COMPLETS, 1):
    if heures() > DUREE_MAX_HEURES + 0.9:
        journal("⏱️ Plus de temps pour d'autres vidéos test.")
        break
    emb = TEXTES[texte]
    t1 = time.time()
    with torch.no_grad():
        images = pipe(prompt_embeds=completer(emb).to(DEVICE, transformer.dtype), negative_prompt_embeds=neg,
                      height=HAUT, width=LARG, num_frames=IMAGES_TEST, num_inference_steps=ETAPES_TEST,
                      guidance_scale=5.0, generator=torch.Generator(device="cpu").manual_seed(42 + i)).frames[0]
    chemin = os.path.join(dossier_videos, f"test_{i}.mp4")
    exporter_video(images, chemin, fps=16)
    journal(f"🎬 Vidéo test {i}/{len(PROMPTS_TEST)} créée en {(time.time() - t1) / 60:.1f} min : {chemin}")

journal(f"🏁 Terminé en {heures():.2f} h. Résultats dans l'onglet Output :")
journal("   • lora_final/pytorch_lora_weights.safetensors  (le module appris)")
journal("   • videos_test/  (vidéos créées)   • apercu_passages_choisis.jpg   • journal.txt")
