// Mannequin IA — fichier unique à coller dans l'éditeur Cloudflare
// Interface de l'app (page affichée sur iPhone). Pas de backticks ni de
// séquence dollar-accolade dans le script ci-dessous.
const PAGE_HTML = `<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0f0f14">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Mannequin IA">
<title>Mannequin IA</title>
<style>
  :root {
    --fond: #0f0f14; --carte: #1a1a22; --carte2: #23232e; --bord: #2f2f3c;
    --texte: #f4f4f7; --doux: #a3a3b5; --accent: #c9a96e; --accent2: #e6cf9f;
    --erreur: #ff7a7a; --ok: #7ad9a3; --rayon: 18px;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { margin: 0; background: var(--fond); color: var(--texte);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif; }
  body { padding: env(safe-area-inset-top) 16px calc(env(safe-area-inset-bottom) + 24px); }
  main { max-width: 560px; margin: 0 auto; }
  header { text-align: center; padding: 28px 0 18px; }
  h1 { font-size: 30px; margin: 0; letter-spacing: -0.5px; font-weight: 700; }
  h1 span { color: var(--accent); }
  .sous-titre { color: var(--doux); margin: 8px 0 0; font-size: 15px; line-height: 1.4; }
  .carte { background: var(--carte); border: 1px solid var(--bord); border-radius: var(--rayon);
    padding: 18px; margin-bottom: 14px; }
  .etape { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 16px; margin: 0 0 14px; }
  .num { width: 26px; height: 26px; border-radius: 50%; background: var(--accent); color: #1a1408;
    display: grid; place-items: center; font-size: 14px; font-weight: 800; flex: none; }
  .zone-photo { display: block; border: 2px dashed var(--bord); border-radius: 14px; padding: 22px;
    text-align: center; cursor: pointer; color: var(--doux); font-size: 15px; position: relative; overflow: hidden; }
  .zone-photo:active { border-color: var(--accent); }
  .zone-photo input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .zone-photo img { display: none; max-width: 100%; max-height: 260px; border-radius: 10px; margin: 0 auto; }
  .zone-photo.remplie { padding: 10px; border-style: solid; }
  .zone-photo.remplie img { display: block; }
  .zone-photo.remplie .invite { display: none; }
  .icone { font-size: 34px; display: block; margin-bottom: 6px; }
  .note { color: var(--doux); font-size: 13px; line-height: 1.45; margin: 10px 0 0; }
  .bascule { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .bascule.trois { grid-template-columns: 1fr 1fr 1fr; }
  .bascule button { min-height: 48px; }
  button { font: inherit; color: var(--texte); background: var(--carte2); border: 1px solid var(--bord);
    border-radius: 12px; padding: 10px 12px; cursor: pointer; transition: background .15s, border-color .15s, transform .1s; }
  button:active { transform: scale(0.98); }
  button.actif { background: rgba(201,169,110,.16); border-color: var(--accent); color: var(--accent2); font-weight: 650; }
  .puces { display: flex; flex-wrap: wrap; gap: 8px; }
  .puces button { font-size: 14px; padding: 9px 13px; border-radius: 999px; min-height: 40px; }
  .libelle { font-size: 13px; color: var(--doux); margin: 16px 0 8px; text-transform: uppercase; letter-spacing: .6px; }
  .libelle:first-of-type { margin-top: 0; }
  .case { display: flex; gap: 12px; align-items: flex-start; font-size: 14px; line-height: 1.45; color: var(--doux); cursor: pointer; }
  .case input { width: 22px; height: 22px; accent-color: var(--accent); flex: none; margin: 0; }
  .principal { width: 100%; min-height: 56px; font-size: 17px; font-weight: 700; border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #1a1408; border-radius: 14px; margin-top: 4px; }
  .principal:disabled { opacity: .45; }
  .compteur { text-align: center; color: var(--doux); font-size: 13px; margin: 10px 0 0; }
  .message { display: none; border-radius: 12px; padding: 12px 14px; font-size: 14px; line-height: 1.45; margin-top: 12px; }
  .message.erreur { display: block; background: rgba(255,122,122,.1); border: 1px solid rgba(255,122,122,.35); color: var(--erreur); }
  #resultat { display: none; }
  #resultat img { width: 100%; border-radius: 14px; display: block; }
  .badge { display: inline-block; font-size: 12px; padding: 4px 10px; border-radius: 999px;
    background: var(--carte2); border: 1px solid var(--bord); color: var(--doux); margin: 12px 0 0; }
  .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
  .actions button { min-height: 50px; font-weight: 650; }
  .chargement { display: none; text-align: center; padding: 26px 10px; }
  .chargement.visible { display: block; }
  .rond { width: 44px; height: 44px; border: 4px solid var(--bord); border-top-color: var(--accent);
    border-radius: 50%; margin: 0 auto 14px; animation: tourne 0.9s linear infinite; }
  @keyframes tourne { to { transform: rotate(360deg); } }
  footer { text-align: center; color: var(--doux); font-size: 12px; line-height: 1.5; margin-top: 18px; padding: 0 8px; }
  @media (prefers-reduced-motion: reduce) { .rond { animation-duration: 2.5s; } button { transition: none; } }
</style>
</head>
<body>
<main>
  <header>
    <h1>Mannequin <span>IA</span></h1>
    <p class="sous-titre">Crée un mannequin fictif inspiré d'une photo, sans jamais copier la personne.</p>
  </header>

  <section class="carte">
    <p class="etape"><span class="num">1</span>Photo d'inspiration</p>
    <label class="zone-photo" id="zonePhoto">
      <input type="file" id="photo" accept="image/*">
      <span class="invite"><span class="icone">📷</span>Touche ici pour choisir une photo</span>
      <img id="apercu" alt="Photo choisie">
    </label>
    <p class="note">L'IA reprend seulement l'ambiance (style, cheveux, allure). Le visage créé est toujours nouveau. Ta photo n'est jamais enregistrée.</p>
  </section>

  <section class="carte">
    <p class="etape"><span class="num">2</span>Ton mannequin</p>
    <p class="libelle">Genre</p>
    <div class="bascule" data-groupe="genre">
      <button type="button" data-valeur="feminin" class="actif">Féminin</button>
      <button type="button" data-valeur="masculin">Masculin</button>
    </div>
    <p class="libelle">Tenue</p>
    <div class="puces" data-groupe="tenue" id="choixTenue"></div>
    <p class="libelle">Décor</p>
    <div class="puces" data-groupe="decor" id="choixDecor"></div>
    <p class="libelle">Cadrage</p>
    <div class="puces" data-groupe="cadrage" id="choixCadrage"></div>
    <p class="libelle">Niveau d'inspiration</p>
    <div class="bascule trois" data-groupe="inspiration">
      <button type="button" data-valeur="legere">Léger</button>
      <button type="button" data-valeur="moyenne" class="actif">Moyen</button>
      <button type="button" data-valeur="forte">Fort</button>
    </div>
    <p class="note">Plus le niveau est fort, plus l'IA reprend de traits (cheveux, silhouette, teint), sans jamais copier le visage.</p>
  </section>

  <section class="carte">
    <p class="etape"><span class="num">3</span>Création</p>
    <label class="case">
      <input type="checkbox" id="certif">
      <span>Je confirme que la photo montre un adulte, et que j'utiliserai les images en respectant les autres (pas de tromperie, pas de harcèlement).</span>
    </label>
    <button type="button" class="principal" id="creer" style="margin-top:16px">Créer mon mannequin</button>
    <p class="compteur" id="compteur"></p>
    <div class="message" id="message" role="alert"></div>

    <div class="chargement" id="chargement">
      <div class="rond"></div>
      <div id="texteChargement">L'IA observe la photo…</div>
    </div>

    <div id="resultat">
      <img id="imageResultat" alt="Mannequin créé par l'IA">
      <span class="badge">✦ Image générée par IA · personne fictive</span>
      <div class="actions">
        <button type="button" id="enregistrer">Enregistrer</button>
        <button type="button" id="recreer">Recréer</button>
      </div>
    </div>
  </section>

  <footer>
    Chaque image contient une mention « générée par IA » dans ses informations, conformément à la loi européenne.<br>
    Les créations ressemblant à une vraie personne, à un mineur ou à du contenu dénudé sont bloquées.
  </footer>
</main>

<script>
(function () {
  "use strict";

  var OPTIONS = {
    tenue: ["Casual chic", "Tenue de soirée", "Streetwear", "Sport", "Costume / Tailleur", "Tenue d'été", "Manteau d'hiver", "Jean & t-shirt"],
    decor: ["Studio fond uni", "Rue de Paris", "Plage", "Nature", "Intérieur luxueux", "Café", "Ville la nuit", "Bureau moderne"],
    cadrage: ["Portrait", "Mi-corps", "En pied", "Assis"]
  };
  var choix = { genre: "feminin", tenue: "Casual chic", decor: "Studio fond uni", cadrage: "Mi-corps", inspiration: "moyenne" };
  var photoDataUrl = null;
  var derniereImage = null;
  var enCours = false;

  function $(id) { return document.getElementById(id); }

  // --- Création des puces de choix -----------------------------------
  [["tenue", "choixTenue"], ["decor", "choixDecor"], ["cadrage", "choixCadrage"]].forEach(function (paire) {
    var conteneur = $(paire[1]);
    OPTIONS[paire[0]].forEach(function (valeur) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = valeur;
      b.setAttribute("data-valeur", valeur);
      if (choix[paire[0]] === valeur) b.className = "actif";
      conteneur.appendChild(b);
    });
  });

  Array.prototype.forEach.call(document.querySelectorAll("[data-groupe]"), function (groupe) {
    groupe.addEventListener("click", function (e) {
      var bouton = e.target.closest("button");
      if (!bouton) return;
      Array.prototype.forEach.call(groupe.querySelectorAll("button"), function (b) { b.classList.remove("actif"); });
      bouton.classList.add("actif");
      choix[groupe.getAttribute("data-groupe")] = bouton.getAttribute("data-valeur");
    });
  });

  // --- Photo : redimensionnée dans le téléphone avant l'envoi ---------
  $("photo").addEventListener("change", function () {
    var fichier = this.files && this.files[0];
    if (!fichier) return;
    cacherMessage();
    var lecteur = new FileReader();
    lecteur.onload = function () {
      var img = new Image();
      img.onload = function () {
        var max = 1024;
        var echelle = Math.min(1, max / Math.max(img.width, img.height));
        var canvas = document.createElement("canvas");
        canvas.width = Math.round(img.width * echelle);
        canvas.height = Math.round(img.height * echelle);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        photoDataUrl = canvas.toDataURL("image/jpeg", 0.88);
        $("apercu").src = photoDataUrl;
        $("zonePhoto").classList.add("remplie");
      };
      img.onerror = function () { afficherErreur("Cette photo n'a pas pu être ouverte. Essaie avec une autre."); };
      img.src = lecteur.result;
    };
    lecteur.readAsDataURL(fichier);
  });

  // --- Compteur d'images restantes ------------------------------------
  function majCompteur(restantes) {
    if (typeof restantes !== "number") return;
    $("compteur").textContent = restantes > 1 ? restantes + " créations restantes aujourd'hui"
      : restantes === 1 ? "1 création restante aujourd'hui" : "Plus de création aujourd'hui, reviens demain !";
  }
  fetch("/api/statut").then(function (r) { return r.json(); }).then(function (d) { majCompteur(d.restantes); }).catch(function () {});

  // --- Création -------------------------------------------------------
  var etapes = ["L'IA observe la photo…", "Elle invente un nouveau visage…", "Mise en place du décor…", "Derniers détails…"];
  var minuteur = null;

  function creer() {
    if (enCours) return;
    cacherMessage();
    if (!photoDataUrl) { afficherErreur("Ajoute d'abord une photo d'inspiration (étape 1)."); return; }
    if (!$("certif").checked) { afficherErreur("Coche la case de confirmation pour continuer."); return; }

    enCours = true;
    $("creer").disabled = true;
    $("resultat").style.display = "none";
    $("chargement").classList.add("visible");
    var i = 0;
    $("texteChargement").textContent = etapes[0];
    minuteur = setInterval(function () { i = Math.min(i + 1, etapes.length - 1); $("texteChargement").textContent = etapes[i]; }, 4000);

    fetch("/api/generer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        photo: photoDataUrl, genre: choix.genre, tenue: choix.tenue, decor: choix.decor,
        cadrage: choix.cadrage, inspiration: choix.inspiration, certification: true
      })
    }).then(function (r) {
      return r.json().catch(function () { return { erreur: "Réponse illisible du serveur." }; });
    }).then(function (d) {
      majCompteur(d.restantes);
      if (d.erreur || !d.image) { afficherErreur(d.erreur || "La création a échoué. Réessaie."); return; }
      derniereImage = d.image;
      $("imageResultat").src = d.image;
      $("resultat").style.display = "block";
      setTimeout(function () { $("resultat").scrollIntoView({ behavior: "smooth", block: "start" }); }, 80);
    }).catch(function () {
      afficherErreur("Pas de connexion internet ou serveur injoignable. Réessaie.");
    }).then(function () {
      clearInterval(minuteur);
      enCours = false;
      $("creer").disabled = false;
      $("chargement").classList.remove("visible");
    });
  }

  $("creer").addEventListener("click", creer);
  $("recreer").addEventListener("click", creer);

  // --- Enregistrer (menu de partage de l'iPhone, sinon téléchargement) -
  $("enregistrer").addEventListener("click", function () {
    if (!derniereImage) return;
    var nom = "mannequin-ia-" + Date.now() + ".jpg";
    fetch(derniereImage).then(function (r) { return r.blob(); }).then(function (blob) {
      var fichier = new File([blob], nom, { type: "image/jpeg" });
      if (navigator.canShare && navigator.canShare({ files: [fichier] })) {
        return navigator.share({ files: [fichier], title: "Mannequin IA" }).catch(function () {});
      }
      var lien = document.createElement("a");
      lien.href = URL.createObjectURL(blob);
      lien.download = nom;
      document.body.appendChild(lien);
      lien.click();
      setTimeout(function () { URL.revokeObjectURL(lien.href); lien.remove(); }, 1000);
    });
  });

  function afficherErreur(texte) { var m = $("message"); m.textContent = texte; m.className = "message erreur"; }
  function cacherMessage() { var m = $("message"); m.textContent = ""; m.className = "message"; }
})();
</script>
</body>
</html>`;

// =====================================================================
//  Mannequin IA — moteur (Cloudflare Worker)
//  Crée des mannequins fictifs féminins ou masculins, INSPIRÉS d'une
//  photo de référence sans jamais la copier.
//
//  Fonctionnement :
//   1. Gemma 3 (vision) décrit la photo avec des MOTS (style, cheveux…),
//      sans jamais décrire les traits précis du visage.
//   2. FLUX.1 [schnell] crée une personne nouvelle à partir de ces mots.
//   => Le visage ne passe jamais de la photo à l'image : il est inventé.
//
//  Sécurité : refus des enfants/adolescents, choix de tenues limités et
//  toujours habillés, filtre anti-nudité de Cloudflare, limite par
//  personne et par jour, marquage « image IA » dans chaque fichier.
//  Aucune photo n'est enregistrée.
// =====================================================================


const MODELE_VISION = "@cf/google/gemma-3-12b-it";
const MODELE_IMAGE = "@cf/black-forest-labs/flux-1-schnell";
const ETAPES_IMAGE = 8;            // qualité maximale de FLUX schnell
const LIMITE_PAR_JOUR = 5;          // images par personne et par jour
const TAILLE_MAX_PHOTO = 4_000_000; // ~3 Mo une fois décodée

// --- Choix proposés (libellé affiché -> description pour l'IA) -------
export const CHOIX = {
  genre: {
    feminin: "woman",
    masculin: "man",
  },
  tenue: {
    "Casual chic": "smart casual outfit, tailored trousers and an elegant knit top",
    "Tenue de soirée": "elegant evening outfit, refined and fully covering",
    "Streetwear": "modern streetwear, oversized hoodie, cargo pants and sneakers",
    "Sport": "athletic sportswear, track jacket and joggers",
    "Costume / Tailleur": "sharp tailored suit",
    "Tenue d'été": "light summer outfit, linen shirt and trousers",
    "Manteau d'hiver": "long wool winter coat with a scarf",
    "Jean & t-shirt": "classic jeans and plain t-shirt",
  },
  decor: {
    "Studio fond uni": "professional photo studio with a seamless neutral backdrop",
    "Rue de Paris": "elegant Parisian street with Haussmann buildings",
    "Plage": "sunny beach promenade",
    "Nature": "lush green park with soft natural light",
    "Intérieur luxueux": "luxurious modern interior",
    "Café": "cozy stylish café",
    "Ville la nuit": "city street at night with bokeh lights",
    "Bureau moderne": "bright modern office",
  },
  cadrage: {
    "Portrait": "close-up head and shoulders portrait",
    "Mi-corps": "medium shot from the waist up",
    "En pied": "full body shot, standing pose",
    "Assis": "seated pose, three-quarter shot",
  },
  inspiration: {
    legere: 1,
    moyenne: 2,
    forte: 3,
  },
};

// Mots interdits : si l'analyse en contient, le trait est ignoré.
const MOTS_INTERDITS = /\b(nude|naked|nsfw|lingerie|underwear|bikini|topless|sexy|sexual|erotic|explicit|child|kid|teen|minor|young girl|young boy|underage)\b/i;

// =====================================================================
export default {
  async fetch(requete, env) {
    const url = new URL(requete.url);
    try {
      if (requete.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
        return new Response(PAGE_HTML, {
          headers: {
            "content-type": "text/html; charset=utf-8",
            "cache-control": "no-cache",
            "x-content-type-options": "nosniff",
            "referrer-policy": "no-referrer",
          },
        });
      }
      if (requete.method === "GET" && url.pathname === "/api/statut") {
        const deja = await lireCompteur(requete, env);
        return json({ restantes: Math.max(0, LIMITE_PAR_JOUR - deja), limite: LIMITE_PAR_JOUR });
      }
      if (requete.method === "POST" && url.pathname === "/api/generer") {
        return await generer(requete, env);
      }
      return json({ erreur: "Page introuvable." }, 404);
    } catch (e) {
      console.error("Erreur inattendue", e);
      return json({ erreur: "Une erreur inattendue est survenue. Réessaie dans un instant." }, 500);
    }
  },
};

// =====================================================================
async function generer(requete, env) {
  if (!env.AI) {
    return json({ erreur: "Le moteur IA n'est pas branché (liaison « AI » manquante)." }, 500);
  }

  // --- 1. Lecture et vérification de la demande ---------------------
  let corps;
  try {
    corps = await requete.json();
  } catch {
    return json({ erreur: "Demande illisible." }, 400);
  }
  const erreurValidation = valider(corps);
  if (erreurValidation) return json({ erreur: erreurValidation }, 400);

  // --- 2. Limite par personne ---------------------------------------
  const deja = await lireCompteur(requete, env);
  if (deja >= LIMITE_PAR_JOUR) {
    return json({
      erreur: "Tu as utilisé tes " + LIMITE_PAR_JOUR + " créations du jour. Reviens demain !",
      restantes: 0,
    }, 429);
  }

  // --- 3. Analyse de la photo (en mots, sans le visage) -------------
  let analyse;
  try {
    analyse = await analyserPhoto(env, corps.photo);
  } catch (e) {
    return erreurIA(e);
  }
  if (!analyse || analyse.person_visible !== true) {
    return json({ erreur: "Je ne vois pas de personne sur cette photo. Essaie avec une photo où une personne est bien visible." }, 422);
  }
  if (analyse.apparent_age !== "adult") {
    return json({ erreur: "Pour protéger les mineurs, seules les photos d'adultes sont acceptées." }, 422);
  }

  // --- 4. Création de l'image ---------------------------------------
  const prompt = construirePrompt(corps, analyse);
  let imageBase64;
  try {
    const sortie = await env.AI.run(MODELE_IMAGE, { prompt, steps: ETAPES_IMAGE });
    imageBase64 = sortie && sortie.image;
    if (!imageBase64) throw new Error("Réponse vide du modèle d'image");
  } catch (e) {
    return erreurIA(e);
  }

  // --- 5. Marquage « image générée par IA » (obligation légale UE) --
  const imageMarquee = ajouterMarquageIA(imageBase64);

  // --- 6. Mise à jour du compteur (seulement si réussite) -----------
  await incrementerCompteur(requete, env, deja);

  return json({
    image: "data:image/jpeg;base64," + imageMarquee,
    restantes: Math.max(0, LIMITE_PAR_JOUR - deja - 1),
  });
}

// =====================================================================
export function valider(c) {
  if (!c || typeof c !== "object") return "Demande vide.";
  if (typeof c.photo !== "string" || !/^data:image\/(jpeg|png|webp);base64,/.test(c.photo)) {
    return "Ajoute une photo de référence (JPEG ou PNG).";
  }
  if (c.photo.length > TAILLE_MAX_PHOTO) return "La photo est trop lourde.";
  if (!(c.genre in CHOIX.genre)) return "Choisis Féminin ou Masculin.";
  if (!(c.tenue in CHOIX.tenue)) return "Choisis une tenue.";
  if (!(c.decor in CHOIX.decor)) return "Choisis un décor.";
  if (!(c.cadrage in CHOIX.cadrage)) return "Choisis un cadrage.";
  if (!(c.inspiration in CHOIX.inspiration)) return "Choisis un niveau d'inspiration.";
  if (c.certification !== true) return "Coche la case de confirmation.";
  return null;
}

// =====================================================================
async function analyserPhoto(env, photo) {
  const consigne =
    "You analyse a reference photo for a fashion app that creates NEW, FICTIONAL people inspired by it. " +
    "Never identify or name anyone. Never describe precise facial features (eye shape, nose, lips, jaw, moles, scars). " +
    "Only describe general, non-identifying traits. Reply with ONLY a JSON object, no other text, using exactly these keys:\n" +
    '{"person_visible": true or false, ' +
    '"apparent_age": "adult" or "teen" or "child" or "unclear", ' +
    '"hair": "colour, length and texture, max 8 words", ' +
    '"skin_tone": "general skin tone, max 4 words", ' +
    '"build": "general body build, max 4 words", ' +
    '"style": "overall fashion style, max 8 words", ' +
    '"vibe": "mood and attitude, max 6 words", ' +
    '"details": "makeup, beard or accessories, max 8 words"}\n' +
    'If you are not sure the person is at least 18, use "unclear".';

  const reponse = await env.AI.run(MODELE_VISION, {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: consigne },
          { type: "image_url", image_url: { url: photo } },
        ],
      },
    ],
    max_tokens: 300,
    temperature: 0.2,
  });
  return lireJSON(reponse && reponse.response);
}

export function lireJSON(texte) {
  if (typeof texte === "object" && texte !== null) return texte;
  if (typeof texte !== "string") return null;
  const debut = texte.indexOf("{");
  const fin = texte.lastIndexOf("}");
  if (debut === -1 || fin <= debut) return null;
  try {
    return JSON.parse(texte.slice(debut, fin + 1));
  } catch {
    return null;
  }
}

// Nettoie un trait renvoyé par l'IA avant de le mettre dans le prompt.
function trait(valeur) {
  if (typeof valeur !== "string") return "";
  const propre = valeur.replace(/[^\p{L}\p{N} ,.'-]/gu, " ").replace(/\s+/g, " ").trim().slice(0, 70);
  if (!propre || MOTS_INTERDITS.test(propre)) return "";
  if (/^(none|n\/a|unknown|not visible|unclear)$/i.test(propre)) return "";
  return propre;
}

export function construirePrompt(c, a) {
  const niveau = CHOIX.inspiration[c.inspiration];
  const traits = [];
  // Inspiration légère : style et ambiance seulement.
  if (trait(a.style)) traits.push("fashion style: " + trait(a.style));
  if (trait(a.vibe)) traits.push("attitude: " + trait(a.vibe));
  // Inspiration moyenne : + cheveux et silhouette.
  if (niveau >= 2) {
    if (trait(a.hair)) traits.push("hair: " + trait(a.hair));
    if (trait(a.build)) traits.push("build: " + trait(a.build));
  }
  // Inspiration forte : + teint et détails.
  if (niveau >= 3) {
    if (trait(a.skin_tone)) traits.push("skin tone: " + trait(a.skin_tone));
    if (trait(a.details)) traits.push(trait(a.details));
  }

  return [
    "Professional fashion photography of a fictional adult " + CHOIX.genre[c.genre] + " model in their twenties or thirties",
    "with an entirely original, unique face",
    traits.join(", "),
    "wearing " + CHOIX.tenue[c.tenue] + ", fully clothed",
    CHOIX.cadrage[c.cadrage],
    "setting: " + CHOIX.decor[c.decor],
    "photorealistic, natural skin texture, sharp focus, soft professional lighting, 85mm lens, high detail, magazine quality",
  ].filter(Boolean).join(". ") + ".";
}

// =====================================================================
//  Marquage IA : ajoute dans le JPEG des métadonnées XMP standard (IPTC)
//  indiquant « contenu créé par un algorithme entraîné ». Invisible pour
//  l'utilisateur, lisible par les plateformes et les logiciels.
// =====================================================================
const XMP_IA =
  '<?xpacket begin="\uFEFF" id="W5M0MpCehiHzreSzNTczkc9d"?>' +
  '<x:xmpmeta xmlns:x="adobe:ns:meta/">' +
  '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">' +
  '<rdf:Description rdf:about=""' +
  ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"' +
  ' xmlns:dc="http://purl.org/dc/elements/1.1/"' +
  ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"' +
  ' Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"' +
  ' xmp:CreatorTool="Mannequin IA (FLUX.1 schnell)">' +
  '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">Image générée par intelligence artificielle. Personne fictive.</rdf:li></rdf:Alt></dc:description>' +
  "</rdf:Description></rdf:RDF></x:xmpmeta>" +
  '<?xpacket end="w"?>';

export function ajouterMarquageIA(base64) {
  const octets = base64VersOctets(base64);
  if (octets.length < 4 || octets[0] !== 0xff || octets[1] !== 0xd8) return base64; // pas un JPEG

  const entete = new TextEncoder().encode("http://ns.adobe.com/xap/1.0/\0");
  const xmp = new TextEncoder().encode(XMP_IA);
  const longueur = 2 + entete.length + xmp.length;
  const segment = new Uint8Array(2 + longueur);
  segment[0] = 0xff;
  segment[1] = 0xe1;
  segment[2] = (longueur >> 8) & 0xff;
  segment[3] = longueur & 0xff;
  segment.set(entete, 4);
  segment.set(xmp, 4 + entete.length);

  // On insère après le segment JFIF (APP0) s'il existe, sinon juste après le début.
  let position = 2;
  if (octets[2] === 0xff && octets[3] === 0xe0) {
    position = 4 + ((octets[4] << 8) | octets[5]);
  }
  const resultat = new Uint8Array(octets.length + segment.length);
  resultat.set(octets.subarray(0, position), 0);
  resultat.set(segment, position);
  resultat.set(octets.subarray(position), position + segment.length);
  return octetsVersBase64(resultat);
}

function base64VersOctets(b64) {
  const binaire = atob(b64);
  const octets = new Uint8Array(binaire.length);
  for (let i = 0; i < binaire.length; i++) octets[i] = binaire.charCodeAt(i);
  return octets;
}

function octetsVersBase64(octets) {
  let binaire = "";
  const bloc = 0x8000;
  for (let i = 0; i < octets.length; i += bloc) {
    binaire += String.fromCharCode.apply(null, octets.subarray(i, i + bloc));
  }
  return btoa(binaire);
}

// =====================================================================
//  Compteur par personne (facultatif : nécessite l'espace KV « LIMITES »).
//  L'adresse IP n'est jamais enregistrée en clair : seulement une
//  empreinte chiffrée qui change chaque jour.
// =====================================================================
async function cleCompteur(requete) {
  const ip = requete.headers.get("CF-Connecting-IP") || "inconnu";
  const jour = new Date().toISOString().slice(0, 10);
  const donnees = new TextEncoder().encode("mannequin-ia|" + jour + "|" + ip);
  const empreinte = await crypto.subtle.digest("SHA-256", donnees);
  const hex = Array.from(new Uint8Array(empreinte)).map((o) => o.toString(16).padStart(2, "0")).join("");
  return "j:" + jour + ":" + hex.slice(0, 32);
}

async function lireCompteur(requete, env) {
  if (!env.LIMITES) return 0;
  try {
    const valeur = await env.LIMITES.get(await cleCompteur(requete));
    return parseInt(valeur || "0", 10) || 0;
  } catch {
    return 0;
  }
}

async function incrementerCompteur(requete, env, deja) {
  if (!env.LIMITES) return;
  try {
    await env.LIMITES.put(await cleCompteur(requete), String(deja + 1), { expirationTtl: 60 * 60 * 48 });
  } catch (e) {
    console.error("Compteur indisponible", e);
  }
}

// =====================================================================
export function erreurIA(e) {
  const message = String((e && e.message) || e || "");
  console.error("Erreur IA :", message);
  if (/4006|neuron|daily free allocation|quota/i.test(message)) {
    return json({ erreur: "Le quota gratuit du jour est atteint pour tout le monde. Reviens demain !" }, 503);
  }
  if (/3030|nsfw/i.test(message)) {
    return json({ erreur: "Le filtre de sécurité a bloqué cette création. Essaie une autre tenue ou un autre décor." }, 422);
  }
  if (/5007|5016|agree|license/i.test(message)) {
    return json({ erreur: "Le modèle d'IA doit d'abord être activé dans Cloudflare." }, 503);
  }
  return json({ erreur: "L'IA n'a pas pu répondre. Réessaie dans un instant." }, 502);
}

function json(donnees, statut = 200) {
  return new Response(JSON.stringify(donnees), {
    status: statut,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
