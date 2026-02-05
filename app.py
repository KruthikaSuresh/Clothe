# app.py
# LOOKBOOK AI — Streamlit + Gemini 3 (Full UI)
# ✅ Sidebar: Preferences + Change Image + Gemini model used + Clear cache
# ✅ Main: Upload + Analyze + Gemini Insight
# ✅ Lookbook page: Trend intelligence + Curated shopping (country links + Pinterest) + Celebrity image tiles

import os
import re
import io
import json
import time
import hashlib
import urllib.parse
import itertools
from typing import Optional, List, Dict, Tuple

import streamlit as st
import feedparser
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import requests

# OpenCV optional (Cloud-safe)
try:
    import cv2
except Exception:
    cv2 = None

# Gemini
import google.generativeai as genai
from google.api_core import exceptions


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="LOOKBOOK AI | Global Style Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# SECRETS / ENV
# -----------------------------
load_dotenv()

def _get_secret(name: str) -> str:
    return os.getenv(name) or st.secrets.get(name, "")

API_KEY = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
keys_env = _get_secret("GEMINI_API_KEYS")

API_KEYS: List[str] = []
if keys_env:
    API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
elif API_KEY:
    API_KEYS = [API_KEY.strip()]

HAS_GEMINI = bool(API_KEYS)
KEY_CYCLE = itertools.cycle(API_KEYS) if API_KEYS else None
CURRENT_KEY = next(KEY_CYCLE) if API_KEYS else None

if HAS_GEMINI and CURRENT_KEY:
    genai.configure(api_key=CURRENT_KEY)

# Gemini 3 candidates (avoid gemini-3-flash which often 404s)
TEXT_MODEL_CANDIDATES = [
    os.getenv("GEMINI_TEXT_MODEL", "").strip(),
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",  # fallback if 3-series not enabled
]
VISION_MODEL_CANDIDATES = [
    os.getenv("GEMINI_VISION_MODEL", "").strip(),
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
]
TEXT_MODEL_CANDIDATES = [m for m in TEXT_MODEL_CANDIDATES if m]
VISION_MODEL_CANDIDATES = [m for m in VISION_MODEL_CANDIDATES if m]


# -----------------------------
# STYLES
# -----------------------------
STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Montserrat:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, div { font-family:'Montserrat',sans-serif!important; font-size:16px!important; color:#151515; line-height:1.55!important;}
html, body, .stApp, [data-testid="stMain"], [data-testid="stAppViewContainer"] { background:#f7f7f9!important; }
.hero{ background:linear-gradient(180deg,#0b0b0f 0%,#12121a 100%); color:#fff; padding:42px 30px; border-radius:18px; margin-bottom:18px; text-align:center; box-shadow:0 18px 40px rgba(0,0,0,0.18);}
.hero h1{ font-family:'Cormorant Garamond',serif!important; font-weight:700!important; color:#fff!important; margin:0; font-size:56px!important;}
.hero p{ color:rgba(255,255,255,0.82)!important; margin:10px 0 0 0; font-size:18px!important;}
.card{ border:1px solid rgba(0,0,0,0.08); border-radius:18px; padding:18px; background:#fff; box-shadow:0 10px 30px rgba(0,0,0,0.06);}
.small-muted{ color:rgba(0,0,0,0.58)!important; font-size:13px!important;}
[data-testid="stSidebar"]{ background:#f2f3f6!important; border-right:1px solid rgba(0,0,0,0.08);}
[data-testid="stSidebar"]{ height:100vh!important; min-height:100vh!important; position:sticky!important; top:0!important; align-self:flex-start; }
[data-testid="stSidebar"] > div:first-child{ max-height:100vh!important; height:100vh!important; overflow-y:auto!important; padding:16px 14px 24px 14px!important; box-sizing:border-box;}
.stButton>button, .stLinkButton>a { border-radius:14px!important; padding:0.65rem 0.9rem!important; font-weight:700!important;}
.chip-wrap{ display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start;}
.chip{ display:flex; flex-direction:column; align-items:center; width:96px;}
.swatch{ border-radius:18px; border:1px solid rgba(0,0,0,0.10); box-shadow:0 4px 14px rgba(0,0,0,0.10);}
.chip-label{ margin-top:8px; font-size:13px!important; font-weight:800; text-align:center;}
.chip-hex{ margin-top:2px; font-size:12px!important; opacity:0.68; text-align:center;}
.pill-row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center;}
.pill{ display:inline-block; padding:6px 10px; border-radius:999px; background:#f2f3f6; border:1px solid rgba(0,0,0,0.10); font-weight:800; font-size:13px!important;}
.pill.small{ padding:4px 8px; font-size:12px!important; opacity:0.92;}
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)


# -----------------------------
# STATE / ROUTES
# -----------------------------
if "view" not in st.session_state:
    st.session_state.view = "upload"

def navigate_to(view_name: str):
    st.session_state.view = view_name
    st.rerun()


# -----------------------------
# HELPERS
# -----------------------------
def _enc(s: str) -> str:
    return urllib.parse.quote_plus(str(s).strip())

def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def save_uploaded_file_to_state(uploaded_file):
    if uploaded_file is None:
        return
    b = uploaded_file.getvalue()
    st.session_state["img_bytes"] = b
    st.session_state["img_name"] = uploaded_file.name
    st.session_state["img_hash"] = hashlib.md5(b).hexdigest()

def get_image_from_state() -> Optional[Image.Image]:
    b = st.session_state.get("img_bytes")
    if not b:
        return None
    return Image.open(io.BytesIO(b)).convert("RGB")

def _safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n\t")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except Exception:
            return None
    return None

def pills_html(items: List[str]) -> str:
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not items:
        return ""
    return "<div class='pill-row'>" + "".join([f"<span class='pill small'>{urllib.parse.unquote_plus(_enc(i))}</span>" for i in items]) + "</div>"

def palette_chips_html(pal: List[dict], size_px: int = 36) -> str:
    if not pal:
        return ""
    parts = []
    for p in pal[:10]:
        cname = (str(p.get("name", "Color")).strip() or "Color").replace("'", "")
        chex = (str(p.get("hex", "#DDDDDD")).strip() or "#DDDDDD").replace("'", "")
        parts.append(
            f"<div class='chip'>"
            f"<div class='swatch' style='width:{size_px}px;height:{size_px}px;background:{chex};'></div>"
            f"<div class='chip-label'>{cname}</div>"
            f"<div class='chip-hex'>{chex}</div>"
            f"</div>"
        )
    return "<div class='chip-wrap'>" + "".join(parts) + "</div>"

def google_images_url(q: str) -> str:
    return f"https://www.google.com/search?tbm=isch&q={_enc(q)}"

def google_shopping_url(q: str) -> str:
    return f"https://www.google.com/search?tbm=shop&q={_enc(q)}"

def pinterest_url(q: str) -> str:
    return f"https://www.pinterest.com/search/pins/?q={_enc(q)}"

def retailer_search_url(base: str, q: str) -> str:
    # Some retailers accept query params; for simplicity, use Google site search when unknown
    if "{q}" in base:
        return base.format(q=_enc(q))
    return f"https://www.google.com/search?q=site:{_enc(base)}+{_enc(q)}"


# -----------------------------
# COUNTRY RETAILERS
# -----------------------------
COUNTRIES = {
    "United States": {
        "gl": "US",
        "retailers": {
            "Amazon": "amazon.com",
            "ASOS": "asos.com",
            "Zara": "zara.com/us",
            "H&M": "hm.com/us",
            "Nordstrom": "nordstrom.com",
        },
    },
    "India": {
        "gl": "IN",
        "retailers": {
            "Myntra": "myntra.com",
            "Ajio": "ajio.com",
            "Nykaa Fashion": "nykaafashion.com",
            "H&M": "hm.com/in",
            "Zara": "zara.com/in",
        },
    },
    "United Kingdom": {
        "gl": "GB",
        "retailers": {
            "ASOS": "asos.com",
            "Zara": "zara.com/uk",
            "H&M": "hm.com/gb",
            "John Lewis": "johnlewis.com",
            "Selfridges": "selfridges.com",
        },
    },
}


# -----------------------------
# CELEBS
# -----------------------------
CELEB_DB = {
    "United States": {
        "Women": ["Zendaya", "Hailey Bieber", "Rihanna"],
        "Men": ["Ryan Gosling", "Harry Styles", "Donald Glover"],
    },
    "India": {
        "Women": ["Deepika Padukone", "Priyanka Chopra", "Alia Bhatt"],
        "Men": ["Ranveer Singh", "Hrithik Roshan", "Shah Rukh Khan"],
    },
    "United Kingdom": {
        "Women": ["Dua Lipa", "Florence Pugh", "Emma Watson"],
        "Men": ["Idris Elba", "Tom Hardy", "David Beckham"],
    },
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_celebrity_image_url(name: str) -> Optional[str]:
    """
    Tries Wikipedia pageimages first. If it fails, returns None.
    """
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "thumbnail",
            "pithumbsize": 600,
            "titles": name,
            "redirects": 1,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = page.get("thumbnail", {})
            if "source" in thumb:
                return thumb["source"]
    except Exception:
        return None
    return None


# -----------------------------
# TREND FEED
# -----------------------------
@st.cache_data(ttl=3600)
def get_country_context(country: str, gender: str):
    cfg = COUNTRIES.get(country, COUNTRIES["United States"])
    gl = cfg["gl"]
    rss_url = (
        f"https://news.google.com/rss/search?q={_enc('fashion street style trends')}"
        f"&hl=en&gl={gl}&ceid={gl}:en"
    )

    headlines = []
    try:
        feed = feedparser.parse(rss_url)
        headlines = list(dict.fromkeys([e.title for e in feed.entries]))[:7]
    except Exception:
        headlines = []

    if not headlines:
        headlines = ["Minimal layering", "Vintage revival", "Tailored silhouettes"]

    celebs = (CELEB_DB.get(country, {}).get(gender) or CELEB_DB["United States"][gender])[:6]
    return headlines, celebs


# -----------------------------
# PALETTE / OFFLINE ANALYSIS
# -----------------------------
FASHION_SWATCHES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Black", (18, 18, 18)),
    ("Charcoal", (54, 54, 58)),
    ("Slate", (88, 96, 110)),
    ("Gray", (152, 156, 162)),
    ("Ivory", (242, 238, 228)),
    ("Cream", (232, 231, 220)),
    ("Beige", (216, 198, 168)),
    ("Tan", (196, 160, 118)),
    ("Camel", (193, 154, 107)),
    ("Chocolate", (82, 54, 42)),
    ("Navy", (20, 36, 74)),
    ("Denim", (55, 88, 132)),
    ("Olive", (96, 106, 58)),
    ("Forest", (24, 68, 44)),
    ("Burgundy", (92, 30, 44)),
    ("Rust", (168, 82, 52)),
    ("Mustard", (199, 164, 50)),
    ("Blush", (222, 170, 170)),
    ("Lavender", (160, 140, 190)),
    ("Teal", (34, 128, 122)),
    ("White", (248, 248, 248)),
]

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _nearest_fashion_name(rgb: Tuple[int, int, int]) -> str:
    def dist2(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2
    best = min(FASHION_SWATCHES, key=lambda sw: dist2(rgb, sw[1]))
    return best[0]

def _rgb_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [x/255.0 for x in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def _dominant_palette(img: Image.Image, k: int = 5) -> List[Dict[str, str]]:
    im = img.copy()
    im.thumbnail((420, 420))
    q = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=10)
    palette = q.getpalette()
    color_counts = q.getcolors() or []
    color_counts.sort(reverse=True, key=lambda x: x[0])

    picked = []
    used_hex = set()
    used_name = set()

    for _, idx in color_counts:
        r = palette[idx*3+0]
        g = palette[idx*3+1]
        b = palette[idx*3+2]
        rgb = (r, g, b)
        hx = _rgb_to_hex(rgb)
        if hx in used_hex:
            continue
        name = _nearest_fashion_name(rgb)
        if name.lower() in used_name:
            continue
        used_hex.add(hx)
        used_name.add(name.lower())
        picked.append({"name": name, "hex": hx})
        if len(picked) >= k:
            break

    while len(picked) < k:
        picked.append({"name": "Cream", "hex": "#E8E7DC"})
    return picked[:k]

def detect_face_box(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None

def sample_skin_rgb(img: Image.Image, face_box: Tuple[int, int, int, int]) -> Tuple[Optional[Tuple[int, int, int]], float]:
    if cv2 is None:
        return None, 0.0
    x, y, w, h = face_box
    rgb = np.array(img.convert("RGB"))
    H, W, _ = rgb.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x+w), min(H, y+h)
    face = rgb[y0:y1, x0:x1]
    if face.size == 0:
        return None, 0.0

    fh, fw, _ = face.shape
    roi = face[int(fh*0.25):int(fh*0.80), int(fw*0.18):int(fw*0.82)]
    if roi.size == 0:
        return None, 0.0

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    skin_mask = (Cr > 135) & (Cr < 180) & (Cb > 85) & (Cb < 135) & (Y > 40)

    if skin_mask.any():
        skin_pixels = roi[skin_mask]
        lum = (0.2126*skin_pixels[:,0] + 0.7152*skin_pixels[:,1] + 0.0722*skin_pixels[:,2])
        lo, hi = np.percentile(lum, [5, 95])
        keep = (lum >= lo) & (lum <= hi)
        skin_pixels = skin_pixels[keep] if keep.any() else skin_pixels
        mean = tuple(int(x) for x in skin_pixels.mean(axis=0))
        return mean, float(skin_mask.mean())

    return None, float(skin_mask.mean())

def classify_undertone_and_depth(mean_rgb: Tuple[int, int, int]) -> Tuple[str, str]:
    if cv2 is None:
        lum = _rgb_luminance(mean_rgb)
        depth = "Light" if lum > 0.66 else "Medium" if lum > 0.50 else "Deep"
        return "Neutral", depth

    rgb_np = np.uint8([[list(mean_rgb)]])
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)[0, 0]
    a, b = float(lab[1]), float(lab[2])

    if b >= 150 and a < 150:
        undertone = "Warm"
    elif b <= 135 and a >= 145:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    lum = _rgb_luminance(mean_rgb)
    if lum >= 0.78:
        depth = "Fair"
    elif lum >= 0.66:
        depth = "Light"
    elif lum >= 0.50:
        depth = "Medium"
    else:
        depth = "Deep"

    return undertone, depth

def offline_style_profile(img: Image.Image, category: str, seed: str) -> dict:
    pal = _dominant_palette(img, k=5)
    names = [p["name"].lower() for p in pal]
    dark = sum(n in ("black", "charcoal", "navy", "chocolate", "forest") for n in names)
    light = sum(n in ("white", "ivory", "cream", "beige", "tan") for n in names)

    if dark >= 3 and light <= 1:
        label = "Modern Minimal"
        ideal = "Clean lines, tailored fit"
    elif light >= 3:
        label = "Quiet Luxury"
        ideal = "Polished silhouettes, refined layering"
    else:
        label = "Smart Casual"
        ideal = "Relaxed structure, balanced proportions"

    return {
        "label": label,
        "ideal_cut": ideal,
        "expert_palette": pal,
        "analysis_mode": "Outfit",
        "complexion": {"undertone": "", "depth": ""},
        "morning_palette": pal,
        "evening_palette": pal,
    }

def offline_complexion_profile(img: Image.Image, category: str, seed: str) -> dict:
    face = detect_face_box(img)
    if not face:
        base = offline_style_profile(img, category, seed)
        base["analysis_mode"] = "Outfit (fallback)"
        base["complexion_note"] = "No face detected — using outfit colors instead."
        return base

    mean_rgb, skin_ratio = sample_skin_rgb(img, face)
    if mean_rgb is None or skin_ratio < 0.02:
        base = offline_style_profile(img, category, seed)
        base["analysis_mode"] = "Outfit (fallback)"
        base["complexion_note"] = "Could not sample skin confidently — using outfit colors instead."
        return base

    undertone, depth = classify_undertone_and_depth(mean_rgb)
    pal = _dominant_palette(img, k=8)

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    scored = [(_rgb_luminance(hex_to_rgb(p["hex"])), p) for p in pal]
    scored.sort(key=lambda t: t[0], reverse=True)
    morning = [p for _, p in scored[:5]]
    evening = [p for _, p in scored[-5:]][::-1]

    return {
        "label": f"{undertone} undertone · {depth} depth",
        "ideal_cut": "Use your palette for outfits, celebrity inspo, Pinterest boards, and shopping searches.",
        "expert_palette": pal[:5],
        "morning_palette": morning[:5],
        "evening_palette": evening[:5],
        "analysis_mode": "Complexion (recommended)",
        "complexion": {"undertone": undertone, "depth": depth, "skin_rgb": mean_rgb, "skin_ratio": skin_ratio},
    }


# -----------------------------
# GEMINI CALLS (quiet, robust)
# -----------------------------
def _rotate_key_if_possible():
    global CURRENT_KEY
    if KEY_CYCLE and len(API_KEYS) > 1:
        CURRENT_KEY = next(KEY_CYCLE)
        genai.configure(api_key=CURRENT_KEY)

def _is_not_found(err: Exception) -> bool:
    s = f"{type(err).__name__}: {err}"
    return ("NotFound" in s) or ("404" in s) or ("is not found" in s) or ("not supported" in s)

def generate_text_with_retry(prompt: str) -> Optional[str]:
    if not HAS_GEMINI:
        return None
    last_err: Optional[Exception] = None

    for model_name in TEXT_MODEL_CANDIDATES:
        for _ in range(max(2, len(API_KEYS))):
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                st.session_state["gemini_text_model_used"] = model_name
                return getattr(resp, "text", None) or None
            except exceptions.ResourceExhausted as e:
                last_err = e
                _rotate_key_if_possible()
                time.sleep(0.8)
            except Exception as e:
                last_err = e
                if _is_not_found(e):
                    break
                time.sleep(0.25)

    st.session_state["gemini_last_error"] = f"{type(last_err).__name__}: {last_err}" if last_err else "Unknown error"
    return None

def generate_multimodal_with_retry(prompt: str, img: Image.Image) -> Optional[str]:
    if not HAS_GEMINI:
        return None
    last_err: Optional[Exception] = None

    for model_name in VISION_MODEL_CANDIDATES:
        for _ in range(max(2, len(API_KEYS))):
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={"response_mime_type": "application/json"},
                )
                resp = model.generate_content([prompt, img.convert("RGB")])
                txt = getattr(resp, "text", None) or None
                st.session_state["gemini_vision_model_used"] = model_name
                return txt
            except exceptions.ResourceExhausted as e:
                last_err = e
                _rotate_key_if_possible()
                time.sleep(0.8)
            except Exception as e:
                last_err = e
                if _is_not_found(e):
                    break
                time.sleep(0.25)

    st.session_state["gemini_last_error"] = f"{type(last_err).__name__}: {last_err}" if last_err else "Unknown error"
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def gemini_image_style_insight_cached(img_hash: str, prompt: str, img_bytes: bytes) -> Optional[dict]:
    if not HAS_GEMINI or not img_bytes:
        return None
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    txt = generate_multimodal_with_retry(prompt, img)
    return _safe_json_loads(txt or "")

def gemini_image_style_insight(style: dict, country: str, category: str) -> Optional[dict]:
    if not HAS_GEMINI:
        return None

    comp = (style.get("complexion") or {})
    undertone = comp.get("undertone") or ""
    depth = comp.get("depth") or ""
    label = style.get("label") or ""

    pal = [p.get("name") for p in (style.get("expert_palette") or []) if p.get("name")]
    morning = [p.get("name") for p in (style.get("morning_palette") or []) if p.get("name")]
    evening = [p.get("name") for p in (style.get("evening_palette") or []) if p.get("name")]

    prompt = f"""
Return STRICT JSON only (no markdown).

Analyze the photo and return JSON with keys:
- outfit_summary: string (1-2 sentences)
- formality: one of ["casual","smart casual","work","evening","formal"]
- style_keywords: array of 6-10 tags
- clothing_items: array of 3-6 clothing pieces (ONLY clothing)
- accessory_items: array of 2-6 accessories (jewelry/bag/shoes)
- recommended_shop_keywords: array of 8 search queries (no brand names) for {category} in {country}
- palette_advice: {{ "morning_best": [..], "evening_best":[..], "avoid":[..] }}
- color_pairings: array of 3 pairings like "Camel + Cream + Gold"
- explanation: 2-3 sentences

Context:
- undertone={undertone}, depth={depth}, label={label}
- candidate_best_colors={pal[:10]}
- candidate_morning_colors={morning[:8]}
- candidate_evening_colors={evening[:8]}
"""
    img_hash = st.session_state.get("img_hash", "img")
    img_bytes = st.session_state.get("img_bytes", b"")
    return gemini_image_style_insight_cached(img_hash, prompt, img_bytes)

def gemini_lookbook_text(country: str, category: str, style: dict, headlines: List[str], celebs: List[str]) -> dict:
    prompt = f"""
Return ONLY valid JSON (no markdown).

Country: {country}
Collection: {category}
Style: {json.dumps(style)}
Trend headlines: {headlines}
Celebrities: {celebs}

Schema:
{{
  "trend_summary": "string",
  "style_translation": ["string","string","string"],
  "outfit_idea": "string",
  "shop_keywords": ["string","string","string","string","string","string"],
  "celeb_styling": [{{"name":"string","wearing":"string"}}]
}}
"""
    txt = generate_text_with_retry(prompt) or ""
    j = _safe_json_loads(txt)
    if not j:
        return {
            "trend_summary": f"Across {country}, trends include: {', '.join(headlines[:3])}.",
            "style_translation": ["Anchor looks in your palette.", "Prioritize clean structure.", "Let texture do the work."],
            "outfit_idea": "Crisp top + tailored bottom + sleek layer.",
            "shop_keywords": ["tailored blazer", "straight trousers", "minimal sneakers", "structured coat", "knit sweater", "leather boots"],
            "celeb_styling": [{"name": c, "wearing": "Use your palette + clean tailoring."} for c in celebs[:5]],
        }
    return j


# -----------------------------
# UI: Upload Screen
# -----------------------------
def render_center_upload_panel():
    st.markdown("## Step 1 — Upload a photo")
    st.caption("Best results: face visible, natural daylight. JPG/PNG/WebP works great.")

    uploaded = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader_center",
    )

    if uploaded is not None:
        st.image(uploaded, caption="Preview", use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        use = st.button("✅ Use this upload", use_container_width=True, disabled=(uploaded is None))
    with c2:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        for k in [
            "img_bytes","img_name","img_hash","style","lookbook","gemini_insight","gemini_last_error",
            "gemini_text_model_used","gemini_vision_model_used"
        ]:
            st.session_state.pop(k, None)
        st.rerun()

    if use and uploaded is not None:
        save_uploaded_file_to_state(uploaded)
        for k in ["style","lookbook","gemini_insight","gemini_last_error","gemini_text_model_used","gemini_vision_model_used"]:
            st.session_state.pop(k, None)
        st.rerun()

def render_upload_screen():
    st.markdown(
        """
        <div class="hero">
            <h1>LOOKBOOK AI</h1>
            <p>Global Style Intelligence · Personalized Curation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    img = get_image_from_state()
    has_img = img is not None

    # Sidebar
    with st.sidebar:
        st.markdown("### Preferences")
        category = st.radio("Collection", ["Women", "Men"], horizontal=True, key="sb_category")
        country = st.selectbox("Region", list(COUNTRIES.keys()), key="sb_country")
        st.session_state["category"] = category
        st.session_state["country"] = country

        # ✅ Change image (requested)
        st.markdown("---")
        st.markdown("### 🖼️ Image")
        new_file = st.file_uploader(
            "Change image",
            type=["jpg", "jpeg", "png", "webp"],
            key="sidebar_image_uploader",
        )
        c_img1, c_img2 = st.columns(2)
        with c_img1:
            use_new = st.button("✅ Use", use_container_width=True, disabled=(new_file is None))
        with c_img2:
            clear_img = st.button("🗑️ Clear", use_container_width=True)

        if clear_img:
            for k in [
                "img_bytes","img_name","img_hash","style","lookbook","gemini_insight","gemini_last_error",
                "gemini_text_model_used","gemini_vision_model_used"
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        if use_new and new_file is not None:
            save_uploaded_file_to_state(new_file)
            for k in ["style","lookbook","gemini_insight","gemini_last_error","gemini_text_model_used","gemini_vision_model_used"]:
                st.session_state.pop(k, None)
            st.rerun()

        # ✅ Only show model used (clean)
        st.markdown("---")
        used_v = st.session_state.get("gemini_vision_model_used")
        used_t = st.session_state.get("gemini_text_model_used")
        if HAS_GEMINI and (used_v or used_t):
            st.success(f"Gemini model used: {used_v or used_t}")
        elif HAS_GEMINI:
            st.info("Gemini: connected")
        else:
            st.warning("Gemini: not connected (missing API key)")

        if st.button("🧹 Clear cache", use_container_width=True):
            st.cache_data.clear()
            for k in ["gemini_insight","gemini_last_error","gemini_text_model_used","gemini_vision_model_used","lookbook"]:
                st.session_state.pop(k, None)
            st.rerun()

    if not has_img:
        render_center_upload_panel()
        return

    st.markdown(
        f"<div class='card' style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div><b>Selected</b> · {st.session_state.get('category','Women')} · {st.session_state.get('country','United States')}</div>"
        f"<div class='small-muted'>Analysis: Complexion (recommended)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    def _run_analysis():
        seed = st.session_state.get("img_hash", "seed")
        st.session_state["style"] = offline_complexion_profile(img, st.session_state["category"], seed)

        for k in ["gemini_insight","gemini_last_error","gemini_text_model_used","gemini_vision_model_used","lookbook"]:
            st.session_state.pop(k, None)

        if HAS_GEMINI:
            with st.spinner("Gemini is analyzing your photo…"):
                insight = gemini_image_style_insight(
                    st.session_state["style"],
                    st.session_state["country"],
                    st.session_state["category"]
                )
            if insight:
                st.session_state["gemini_insight"] = insight
            else:
                st.session_state["gemini_last_error"] = st.session_state.get("gemini_last_error") or "Gemini failed."
        else:
            st.session_state["gemini_last_error"] = "No Gemini key."

        st.rerun()

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Your Upload")
        st.caption(st.session_state.get("img_name", "uploaded image"))
        st.image(img, use_container_width=True)
        if st.button("✨ Analyze style", type="primary", use_container_width=True, key="analyze_left"):
            _run_analysis()
        st.markdown("</div>", unsafe_allow_html=True)

        style = st.session_state.get("style")
        if style:
            st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.subheader("Your palette")
            st.markdown(palette_chips_html(style.get("expert_palette", [])), unsafe_allow_html=True)

            mp = style.get("morning_palette") or []
            ep = style.get("evening_palette") or []
            if mp and ep:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.write("Morning palette")
                st.markdown(palette_chips_html(mp, size_px=30), unsafe_allow_html=True)
                st.write("Evening palette")
                st.markdown(palette_chips_html(ep, size_px=30), unsafe_allow_html=True)

            note = style.get("complexion_note")
            if note:
                st.caption(note)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.button("✨ Analyze style", type="primary", use_container_width=True, key="analyze_right"):
            _run_analysis()

        insight = st.session_state.get("gemini_insight")

        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        with st.expander("✨ Gemini Insight (style + color reasoning)", expanded=True):
            if insight:
                st.write("**Outfit summary**")
                st.caption(insight.get("outfit_summary", ""))

                if insight.get("style_keywords"):
                    st.write("**Style tags**")
                    st.markdown(pills_html(insight.get("style_keywords", [])[:12]), unsafe_allow_html=True)

                if insight.get("palette_advice"):
                    padv = insight.get("palette_advice") or {}
                    cA, cB, cC = st.columns(3)
                    with cA:
                        st.caption("🌤 Morning best")
                        st.markdown(pills_html((padv.get("morning_best") or [])[:8]), unsafe_allow_html=True)
                    with cB:
                        st.caption("🌙 Evening best")
                        st.markdown(pills_html((padv.get("evening_best") or [])[:8]), unsafe_allow_html=True)
                    with cC:
                        st.caption("🚫 Avoid")
                        st.markdown(pills_html((padv.get("avoid") or [])[:8]), unsafe_allow_html=True)

                if insight.get("color_pairings"):
                    st.write("**Pairings**")
                    st.markdown(pills_html(insight.get("color_pairings", [])[:8]), unsafe_allow_html=True)

                if insight.get("explanation"):
                    st.write("**Why these colors work**")
                    st.caption(insight.get("explanation"))
            else:
                st.caption("Click **Analyze style** to generate Gemini insights.")
        st.markdown("</div>", unsafe_allow_html=True)

        style = st.session_state.get("style")
        if style:
            st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.subheader("🚀 Next step")
            st.caption("Generate a regional lookbook (Gemini text).")
            if st.button("Open Lookbook", use_container_width=True):
                headlines, celebs = get_country_context(st.session_state["country"], st.session_state["category"])
                st.session_state["lookbook"] = gemini_lookbook_text(
                    st.session_state["country"],
                    st.session_state["category"],
                    style,
                    headlines,
                    celebs,
                )
                st.session_state["celebs_for_lookbook"] = celebs
                navigate_to("lookbook")
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# UI: Lookbook Screen (FULL: shopping + celeb images + Pinterest)
# -----------------------------
def render_lookbook_screen():
    country = st.session_state.get("country", "United States")
    category = st.session_state.get("category", "Women")
    style = st.session_state.get("style", {}) or {}
    lb = st.session_state.get("lookbook", {}) or {}

    top = st.columns([1, 2, 1])
    with top[0]:
        if st.button("← Back"):
            navigate_to("upload")

    st.markdown(f"<h2 style='text-align:center; margin-top:10px;'>THE {country.upper()} EDIT</h2>", unsafe_allow_html=True)

    if style.get("expert_palette"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Your palette**")
        st.markdown(palette_chips_html(style.get("expert_palette", []), size_px=30), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Ensure lookbook content exists
    if not lb:
        headlines, celebs = get_country_context(country, category)
        lb = gemini_lookbook_text(country, category, style, headlines, celebs)
        st.session_state["lookbook"] = lb
        st.session_state["celebs_for_lookbook"] = celebs

    celebs = st.session_state.get("celebs_for_lookbook") or (CELEB_DB.get(country, {}).get(category) or [])

    tabs = st.tabs(["📈 Trend intelligence", "🛍️ Curated shopping", "⭐ Celebrity images"])

    # -------- Trend intelligence
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info(lb.get("trend_summary", ""))
        for tip in (lb.get("style_translation", []) or [])[:6]:
            st.write("•", tip)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.subheader("Outfit idea")
        st.write(lb.get("outfit_idea", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.subheader("Celebrity styling")
        for c in (lb.get("celeb_styling") or [])[:6]:
            st.write(f"**{c.get('name','')}** — {c.get('wearing','')}")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Curated shopping
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Curated shopping links")
        st.caption("These links are generated from your palette + lookbook keywords. Choose a color, then open retailers / Pinterest.")

        pal_names = [p.get("name") for p in (style.get("expert_palette") or []) if p.get("name")] or ["Neutral"]
        selected_color = st.selectbox("Suggested color", pal_names, key="shop_color_pick")

        base_keywords = unique_keep_order(lb.get("shop_keywords", []) or [])
        if not base_keywords:
            base_keywords = ["tailored blazer", "straight trousers", "minimal sneakers", "structured coat"]

        # Build 6–8 queries
        queries = []
        for k in base_keywords[:6]:
            queries.append(f"{selected_color} {k} {category}")
        queries = unique_keep_order(queries)

        cfg = COUNTRIES.get(country, COUNTRIES["United States"])
        retailers = cfg.get("retailers", {})

        st.write("**Retailers (opens search results):**")
        for rname, domain in retailers.items():
            q = f"{selected_color} {base_keywords[0]} {category}"
            st.link_button(f"{rname}", retailer_search_url(domain, q), use_container_width=True)

        st.markdown("---")
        st.write("**Pinterest boards (palette-aware):**")
        st.link_button(
            f"Open Pinterest for {selected_color} {category} outfits",
            pinterest_url(f"{selected_color} {category} outfit ideas"),
            use_container_width=True,
        )

        st.markdown("---")
        st.write("**More shopping searches (Google Shopping):**")
        cols = st.columns(2)
        for i, q in enumerate(queries[:8]):
            with cols[i % 2]:
                st.link_button(q, google_shopping_url(q), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Celebrity images
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Celebrity inspiration")
        st.caption("Images load from Wikipedia when available. Use buttons for Google Images + Pinterest boards.")

        pal_names = [p.get("name") for p in (style.get("expert_palette") or []) if p.get("name")] or ["Neutral"]
        pin_color = st.selectbox("Pinterest color filter", pal_names, key="celeb_pin_color")

        grid = st.columns(3)
        for idx, celeb in enumerate(celebs[:9]):
            with grid[idx % 3]:
                img_url = get_celebrity_image_url(celeb)
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    # fallback placeholder (still looks nice)
                    st.image(f"https://source.unsplash.com/featured/600x600?portrait,{_enc(celeb)}", use_container_width=True)

                st.markdown(f"**{celeb}**")
                st.link_button("Google Images", google_images_url(f"{celeb} street style {country}"), use_container_width=True)
                st.link_button("Pinterest", pinterest_url(f"{celeb} {pin_color} outfit"), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# MAIN
# -----------------------------
if st.session_state.view == "upload":
    render_upload_screen()
elif st.session_state.view == "lookbook":
    render_lookbook_screen()
