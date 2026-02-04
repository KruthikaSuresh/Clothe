# app.py
import os
import re
import json
import time
import hashlib
import urllib.parse


def _decode_plus(s: str) -> str:
    try:
        return urllib.parse.unquote(str(s)).replace("+", " + ").replace("  ", " ").strip()
    except Exception:
        return str(s)
import itertools
import io
from typing import Optional, List, Dict, Tuple

import streamlit as st
import feedparser
import requests
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv

# Gemini (text-only in this app because your project has no multimodal quota)
import google.generativeai as genai
from google.api_core import exceptions

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="LOOKBOOK AI | Global Style Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
keys_env = os.getenv("GEMINI_API_KEYS")
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()] if keys_env else []
if API_KEY and not API_KEYS:
    API_KEYS = [API_KEY]

# If no keys, we still run (offline-only), but we won't call Gemini
HAS_GEMINI = bool(API_KEYS)

KEY_CYCLE = itertools.cycle(API_KEYS) if API_KEYS else None
CURRENT_KEY = next(KEY_CYCLE) if API_KEYS else None
if HAS_GEMINI:
    genai.configure(api_key=CURRENT_KEY)

# -----------------------------
# UI THEME (clean + readable)
# -----------------------------
STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Montserrat:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, div {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 16px !important;
    color: #151515;
    line-height: 1.55 !important;
}

/* Force light canvas even if browser/OS is dark (v16 theme) */
html, body { background: #f7f7f9 !important; }
.stApp { background: #f7f7f9 !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f7f7f9 !important; }
.block-container { background: #f7f7f9 !important; padding-bottom: 3rem !important; }
section.main > div { padding-top: 18px; padding-bottom: 24px; }

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 700 !important;
    color: #0c0c0c !important;
}

.hero {
    background: linear-gradient(180deg, #0b0b0f 0%, #12121a 100%);
    color: #fff;
    padding: 42px 30px;
    border-radius: 18px;
    margin-bottom: 18px;
    text-align: center;
    box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}
.hero h1 { color: #fff !important; margin: 0; font-size: 56px !important; letter-spacing: 0.4px; }
.hero p  { color: rgba(255,255,255,0.82) !important; margin: 10px 0 0 0; font-size: 18px !important; }

.card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 18px 18px;
    background: #ffffff;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

.small-muted { color: rgba(0,0,0,0.58) !important; font-size: 13px !important; }

hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 26px 0; }

[data-testid="stSidebar"] {
    background: #f2f3f6 !important;
    border-right: 1px solid rgba(0,0,0,0.08);
}

/* Sidebar polish: prevent the "cut off" look */
[data-testid="stSidebar"] {
    height: 100vh !important;
    min-height: 100vh !important;
    position: sticky !important;
    top: 0 !important;
    align-self: flex-start;
    box-shadow: 10px 0 28px rgba(0,0,0,0.06);
}

/* Add consistent inner padding + ensure sidebar scroll feels natural */
[data-testid="stSidebar"] > div:first-child {
    max-height: 100vh !important;
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    -webkit-overflow-scrolling: touch;
    padding: 16px 14px 24px 14px !important;
    box-sizing: border-box;
}

/* Make the sidebar header area feel less clipped */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}

/* Tighter buttons */
.stButton>button, .stLinkButton>a {
    border-radius: 14px !important;
    padding: 0.65rem 0.9rem !important;
    font-weight: 600 !important;
}
.stLinkButton>a { text-decoration: none !important; }

/* Tabs: cleaner, more compact */
div[data-testid="stTabs"] button {
    font-weight: 700 !important;
    padding: 10px 14px !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    border-bottom: 3px solid rgba(0,0,0,0.85) !important;
}

/* Selectbox: compact + readable (fix "blank pill" look)
   NOTE: Streamlit/Cloud can change testids, so we style BaseWeb "select" directly too.
*/
div[data-baseweb="select"] > div {
    min-height: 42px !important;
    border-radius: 14px !important;
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] div {
    color: #151515 !important;
    font-weight: 650 !important;
}


/* Palette chips */
.chip-wrap{ display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start; }
.chip{ display:flex; flex-direction:column; align-items:center; width:96px; }
.swatch{ border-radius:18px; border:1px solid rgba(0,0,0,0.10); box-shadow: 0 4px 14px rgba(0,0,0,0.10); }
.chip-label{ margin-top:8px; font-size:13px !important; font-weight:700; text-align:center; }
.chip-hex{ margin-top:2px; font-size:12px !important; opacity:0.68; text-align:center; }

/* Pills / badges */
.pill-row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.pill{ display:inline-block; padding:6px 10px; border-radius:999px; background:#f2f3f6; border:1px solid rgba(0,0,0,0.10); font-weight:700; font-size:13px !important; }
.pill.small{ padding:4px 8px; font-size:12px !important; opacity:0.92; }
.badge{ display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(0,0,0,0.06); border:1px solid rgba(0,0,0,0.08); font-weight:800; font-size:13px !important; }
.card-tight{ padding:14px 16px; }


/* --- Two-pane layout: sticky image + normal page scroll ---
   Option A: allow main page scrolling, keep image sticky so the page never looks empty.
   This is the most reliable approach across Safari/Chrome.
*/

/* Allow normal page scrolling */
html, body {
  height: auto;
  overflow: auto !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.stApp {
  height: auto !important;
  overflow: visible !important;
}

/* Sticky left pane */
.sticky-left {
  position: sticky;
  top: 16px;
  align-self: flex-start;
}

</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# -----------------------------
# STATE / ROUTING
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
    # Default encoding for most retailers (spaces as +)
    return urllib.parse.quote_plus(str(s).strip())

def _enc_retailer(retailer: str, s: str) -> str:
    """Retailer-specific encoding.
    Zara search endpoints are picky about '+' for spaces, so we use percent-encoding (%20).
    """
    s = str(s).strip()
    if retailer.lower() == "zara":
        return urllib.parse.quote(s, safe="")
    return urllib.parse.quote_plus(s)


def _inject_gender_hint(query: str, category: Optional[str]) -> str:
    """Ensure inspiration searches are gender-aligned.

    Pinterest often defaults to women-oriented results unless the query explicitly
    includes a gender hint. We inject "men" / "women" (or "mens" / "womens")
    when missing.
    """
    q = str(query or "").strip()
    if not category:
        return q

    cat = str(category).strip().lower()
    q_low = q.lower()

    # If user already specified gender, don't touch it.
    if any(tok in q_low for tok in [" men ", " men's", " mens", " male", " women ", " women's", " womens", " female"]):
        return q

    if cat == "men":
        return f"men {q}".strip()
    if cat == "women":
        return f"women {q}".strip()
    return q


def build_pinterest(query: str, category: Optional[str] = None) -> str:
    """Build a Pinterest search URL for an arbitrary query string.

    If category is provided (Men/Women), we bias the search to that gender.
    """
    q = _inject_gender_hint(query, category)
    return f"https://www.pinterest.com/search/pins/?q={_enc(q)}"


def unique_keep_order(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def save_uploaded_file_to_state(uploaded_file):
    if uploaded_file is None:
        return
    b = uploaded_file.getvalue()
    st.session_state["img_bytes"] = b
    st.session_state["img_name"] = uploaded_file.name
    st.session_state["img_mime"] = uploaded_file.type or "image/jpeg"
    st.session_state["img_hash"] = hashlib.md5(b).hexdigest()


def get_image_from_state() -> Optional[Image.Image]:
    b = st.session_state.get("img_bytes")
    if not b:
        return None
    return Image.open(io.BytesIO(b)).convert("RGB")


def extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n\t")
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    return {}


# -----------------------------
# OFFLINE STYLE + PALETTE (no Gemini needed)
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
    ("Wine", (74, 22, 35)),
    ("Rust", (168, 82, 52)),
    ("Terracotta", (186, 92, 70)),
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
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    best = min(FASHION_SWATCHES, key=lambda sw: dist2(rgb, sw[1]))
    return best[0]


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

    for count, idx in color_counts:
        r = palette[idx * 3 + 0]
        g = palette[idx * 3 + 1]
        b = palette[idx * 3 + 2]
        rgb = (r, g, b)
        hx = _rgb_to_hex(rgb)

        if hx in used_hex:
            continue

        name = _nearest_fashion_name(rgb)

        # IMPORTANT: avoid duplicates by name too (fixes Olive/Gray repeated in dropdown)
        if name.lower() in used_name:
            continue

        used_hex.add(hx)
        used_name.add(name.lower())
        picked.append({"name": name, "hex": hx})

        if len(picked) >= k:
            break

    while len(picked) < k:
        # fill with safe neutrals if needed
        filler = {"name": "Cream", "hex": "#E8E7DC"}
        if filler["name"].lower() not in used_name:
            picked.append(filler)
            used_name.add(filler["name"].lower())
        else:
            picked.append({"name": "Gray", "hex": "#9A9CA2"})
        if len(picked) >= k:
            break

    return picked[:k]



# -----------------------------
# COMPLEXION-BASED COLOR ANALYSIS (face-first)
# -----------------------------
def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except Exception:
        return (220, 220, 220)

def _rgb_luminance(rgb: Tuple[int,int,int]) -> float:
    r,g,b = [x/255.0 for x in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def detect_face_box(img: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    """Return (x,y,w,h) for the largest detected face, or None."""
    try:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        # Built-in OpenCV Haar cascade (ships with cv2)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        if faces is None or len(faces) == 0:
            return None
        # pick largest
        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None

def sample_skin_rgb(img: Image.Image, face_box: Tuple[int,int,int,int]) -> Tuple[Optional[Tuple[int,int,int]], float]:
    """Return (mean_skin_rgb, skin_ratio) within mid-face region."""
    x,y,w,h = face_box
    rgb = np.array(img.convert("RGB"))
    H,W,_ = rgb.shape
    # clamp
    x0 = max(0, x); y0=max(0,y); x1=min(W, x+w); y1=min(H, y+h)
    face = rgb[y0:y1, x0:x1]
    if face.size == 0:
        return None, 0.0

    # focus on mid-face (reduce hair/forehead + chin)
    fh, fw, _ = face.shape
    ry0 = int(fh*0.25); ry1 = int(fh*0.80)
    rx0 = int(fw*0.18); rx1 = int(fw*0.82)
    roi = face[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None, 0.0

    # Skin mask in YCrCb (common heuristic)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:,:,0]; Cr = ycrcb[:,:,1]; Cb = ycrcb[:,:,2]
    skin_mask = (Cr > 135) & (Cr < 180) & (Cb > 85) & (Cb < 135) & (Y > 40)
    # Remove very dark/very bright outliers (shadows/highlights)
    if skin_mask.any():
        skin_pixels = roi[skin_mask]
        # trim extremes
        lum = (0.2126*skin_pixels[:,0] + 0.7152*skin_pixels[:,1] + 0.0722*skin_pixels[:,2])
        lo, hi = np.percentile(lum, [5, 95])
        keep = (lum >= lo) & (lum <= hi)
        skin_pixels = skin_pixels[keep] if keep.any() else skin_pixels
        mean = tuple(int(x) for x in skin_pixels.mean(axis=0))
        ratio = float(skin_mask.mean())
        return mean, ratio
    return None, float(skin_mask.mean())

def classify_undertone_and_depth(mean_rgb: Tuple[int,int,int]) -> Tuple[str, str]:
    """Return (undertone, depth). undertone in {Warm,Cool,Neutral}; depth in {Fair,Light,Medium,Deep}."""
    rgb_np = np.uint8([[list(mean_rgb)]])
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)[0,0]
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])

    # Undertone heuristic: b* relates to yellow/blue; a* to red/green
    # These thresholds are empirical and intentionally conservative.
    if b >= 150 and a < 150:
        undertone = "Warm"
    elif b <= 135 and a >= 145:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    # Depth from luminance of skin sample
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

# Curated flattering palettes (names + hex)
FLATTERING_PALETTES: Dict[str, List[Tuple[str,str]]] = {
    # Cool + Deep (Winter-like)
    "Cool_Deep": [
        ("Crisp White", "#FFFFFF"), ("Charcoal", "#2E2E2E"), ("Black", "#0B0B0F"), ("Navy", "#0B1F3A"),
        ("Cobalt", "#1F4ED8"), ("Emerald", "#0B7A5A"), ("Fuchsia", "#C2185B"), ("Icy Gray", "#DDE1E6"),
        ("Silver", "#C0C0C0"), ("Berry", "#7B1B4A"),
    ],
    # Warm + Deep (Autumn-like)
    "Warm_Deep": [
        ("Cream", "#FFF1D6"), ("Camel", "#C19A6B"), ("Espresso", "#2B1B12"), ("Chocolate", "#4E2A1E"),
        ("Olive", "#556B2F"), ("Teal", "#0E6F6E"), ("Rust", "#B55239"), ("Mustard", "#C8A600"),
        ("Terracotta", "#C0604D"), ("Warm Gray", "#8A817C"),
    ],
    # Cool + Light (Summer-like)
    "Cool_Light": [
        ("Soft White", "#F7F7F2"), ("Dove Gray", "#B8BCC2"), ("Slate", "#6B7785"), ("Dusty Blue", "#6E8FB3"),
        ("Periwinkle", "#8AAAE5"), ("Lavender", "#A78BFA"), ("Rose", "#D18FA6"), ("Sage", "#A3B18A"),
        ("Soft Navy", "#2F3E56"), ("Pewter", "#8E9AAF"),
    ],
    # Warm + Light (Spring-like)
    "Warm_Light": [
        ("Warm Ivory", "#FFF6E3"), ("Sand", "#D9C2A6"), ("Beige", "#D2B48C"), ("Peach", "#F4B59F"),
        ("Coral", "#FF6F61"), ("Apricot", "#F7A35C"), ("Aqua", "#2EC4B6"), ("Warm Green", "#6BBF59"),
        ("Butter Yellow", "#FFD166"), ("Sky Blue", "#77B6EA"),
    ],
    # Neutral fallback (universal)
    "Neutral": [
        ("White", "#FFFFFF"), ("Off-White", "#F3F1EA"), ("Gray", "#7A7A7A"), ("Charcoal", "#2E2E2E"),
        ("Navy", "#0B1F3A"), ("Camel", "#C19A6B"), ("Olive", "#556B2F"), ("Teal", "#0E6F6E"),
        ("Burgundy", "#6D1A36"), ("Chocolate", "#4E2A1E"),
    ],
}

def pick_flattering_palette(undertone: str, depth: str) -> List[Dict[str,str]]:
    key = None
    if undertone in ("Warm","Cool") and depth in ("Fair","Light","Medium","Deep"):
        # bucket light vs deep
        light = depth in ("Fair","Light")
        if undertone == "Cool" and not light:
            key = "Cool_Deep"
        elif undertone == "Warm" and not light:
            key = "Warm_Deep"
        elif undertone == "Cool" and light:
            key = "Cool_Light"
        elif undertone == "Warm" and light:
            key = "Warm_Light"
    if undertone == "Neutral" or key is None:
        key = "Neutral"

    pal = [{"name": n, "hex": h} for (n,h) in FLATTERING_PALETTES.get(key, FLATTERING_PALETTES["Neutral"])]
    return pal

def _darken_hex(hex_color: str, factor: float = 0.72) -> str:
    """Darken a hex color by multiplying RGB by factor (0-1)."""
    try:
        h = str(hex_color).strip().lstrip("#")
        if len(h) != 6:
            return "#333333"
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#333333"


@st.cache_data(ttl=3600, show_spinner=False)
def gemini_day_night_palettes_cached(img_hash: str, undertone: str, depth: str, base_colors: List[Dict[str, str]]) -> Optional[dict]:
    """Ask Gemini (text) to create *distinct* Morning vs Evening palettes from a base palette."""
    if not HAS_GEMINI:
        return None

    base_list = []
    for c in (base_colors or [])[:12]:
        n = (c.get("name") or "").strip()
        h = (c.get("hex") or "").strip()
        if n and h:
            base_list.append({"name": n, "hex": h})

    prompt = f"""You are a color stylist for a Gemini 3 hackathon demo.
Return STRICT JSON only (no markdown). Create TWO DISTINCT palettes for the same person:

Context:
- undertone: {undertone or "unknown"}
- depth: {depth or "unknown"}
- base_palette: {json.dumps(base_list)}

Output keys:
- morning_palette: array of 5 items, each item is {{ "name": <color name>, "hex": <#RRGGBB> }}
- evening_palette: array of 5 items, each item is {{ "name": <color name>, "hex": <#RRGGBB> }}

Rules:
- Morning is for daylight/work: softer/lighter neutrals and medium accents.
- Evening is for low light/night photos: deeper/richer, higher contrast tones.
- Palettes should NOT overlap more than 1 color name.
- Prefer using/deriving from base_palette, but you MAY add new compatible colors if needed.
"""
    txt = generate_text_with_retry(prompt)
    return _safe_json_loads(txt) if txt else None


def split_morning_evening(pal: List[Dict[str,str]], undertone: str = "", depth: str = "", img_hash: str = "") -> Tuple[List[Dict[str,str]], List[Dict[str,str]]]:
    """Create distinct morning/evening palettes.

    Strategy:
    1) If we have enough colors, split by luminance into non-overlapping sets.
    2) If overlap would be high (small palette), ask Gemini (text) to generate distinct palettes.
    3) Final fallback: darken morning colors to synthesize an evening palette.
    """
    if not pal:
        return [], []

    # Normalize + de-dupe by name
    seen = set()
    norm = []
    for p in pal:
        name = (p.get("name") or "").strip()
        hx = (p.get("hex") or "").strip()
        if not name or not hx:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        norm.append({"name": name, "hex": hx})

    if not norm:
        return [], []

    # If palette is small, Gemini is the best way to produce *real* day vs night differences.
    if len(norm) < 8 and HAS_GEMINI and img_hash:
        j = gemini_day_night_palettes_cached(img_hash, undertone, depth, norm) or {}
        mp = j.get("morning_palette") or []
        ep = j.get("evening_palette") or []
        if isinstance(mp, list) and isinstance(ep, list) and len(mp) >= 4 and len(ep) >= 4:
            return mp[:5], ep[:5]

    # Split by luminance (non-overlapping if possible)
    scored = []
    for p in norm:
        rgb = _hex_to_rgb(p.get("hex", "#DDDDDD"))
        scored.append((_rgb_luminance(rgb), p))
    scored.sort(key=lambda t: t[0], reverse=True)

    morning = [p for _, p in scored[:5]]

    remaining = [p for _, p in scored if p not in morning]
    evening = remaining[-5:][::-1] if remaining else []

    # If still too small / overlaps, synthesize evening by darkening morning colors
    if len(evening) < 4:
        evening = []
        for p in morning[:5]:
            evening.append({"name": f"Deep {p['name']}", "hex": _darken_hex(p["hex"], 0.70)})
        # ensure at least 5
        evening = evening[:5]

    # Final guard: keep overlap <= 1 by name
    mnames = {p["name"].strip().lower() for p in morning}
    cleaned_evening = []
    for p in evening:
        if p["name"].strip().lower() in mnames and len(cleaned_evening) < 4:
            continue
        cleaned_evening.append(p)
    if len(cleaned_evening) >= 4:
        evening = cleaned_evening[:5]

    return morning[:5], evening[:5]



def offline_complexion_profile(img: Image.Image, category: str, seed: str) -> dict:
    face = detect_face_box(img)
    if not face:
        # fallback to outfit-based
        base = offline_style_profile(img, category, seed)
        base["analysis_mode"] = "Outfit (fallback)"
        base["complexion_note"] = "No face detected — using outfit colors instead. Upload a selfie/face-visible photo for complexion-based recommendations."
        return base

    mean_rgb, skin_ratio = sample_skin_rgb(img, face)
    if mean_rgb is None or skin_ratio < 0.02:
        base = offline_style_profile(img, category, seed)
        base["analysis_mode"] = "Outfit (fallback)"
        base["complexion_note"] = "Could not confidently sample skin (lighting/angle). Using outfit colors instead. Try a front-facing selfie in daylight."
        return base

    undertone, depth = classify_undertone_and_depth(mean_rgb)
    flattering = pick_flattering_palette(undertone, depth)
    morning, evening = split_morning_evening(flattering)

    # Keep the app's expected keys
    label = f"{undertone} undertone · {depth} depth"
    ideal = "Use your palette for outfits, celebrity inspo, Pinterest boards, and shopping searches."

    return {
        "label": label,
        "ideal_cut": ideal,
        "expert_palette": flattering,   # used everywhere in UI
        "morning_palette": morning,
        "evening_palette": evening,
        "analysis_mode": "Complexion",
        "complexion": {"undertone": undertone, "depth": depth, "skin_rgb": mean_rgb, "skin_ratio": skin_ratio},
        "items": ["Top", "Bottom", "Shoes"],
    }


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
        "items": ["Top", "Bottom", "Shoes"],
    }


# -----------------------------
# GEMINI (TEXT-ONLY) FOR LOOKBOOK
# -----------------------------
def resolve_text_model_name() -> Optional[str]:
    if not HAS_GEMINI:
        return None
    try:
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        # Avoid experimental variants that often have separate quotas
        models = [m for m in models if "-exp" not in m and "experimental" not in m.lower()]

        preferred = ["gemini-2.5-flash", "gemini-3-flash", "gemini-2.5-flash-lite"]
        for want in preferred:
            for m in models:
                if want in m:
                    return m
        return models[0] if models else None
    except Exception:
        return None


def generate_text_with_retry(prompt: str) -> Optional[str]:
    global CURRENT_KEY
    if not HAS_GEMINI:
        return None

    model_name = resolve_text_model_name()
    if not model_name:
        return None

    last_err = None
    for _ in range(max(2, len(API_KEYS) * 2)):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", None) or None
        except exceptions.ResourceExhausted as e:
            last_err = e
            if len(API_KEYS) > 1:
                CURRENT_KEY = next(KEY_CYCLE)
                genai.configure(api_key=CURRENT_KEY)
            time.sleep(0.6)
            continue
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "billing" in msg or "rate" in msg:
                return None
            time.sleep(0.2)
            continue

    _ = last_err
    return None


# -----------------------------
# GEMINI MULTIMODAL (VISION) — HACKATHON INTEGRATION
# -----------------------------
def resolve_multimodal_model_name() -> Optional[str]:
    """Pick a Gemini model that can handle image+text. Preference order is tuned for Gemini 3 hackathons."""
    if not HAS_GEMINI:
        return None
    # Allow override
    override = os.getenv("GEMINI_MODEL")
    if override:
        return override

    try:
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        models = [m for m in models if "-exp" not in m and "experimental" not in m.lower()]

        preferred = [
            "gemini-3",          # hackathon branding often uses Gemini 3
            "gemini-2.5",        # strong default
            "gemini-2.0",
            "gemini-1.5",
        ]
        # Prefer flash-ish models first (speed + quota)
        preferred_flavors = ["flash", "flash-lite", "flash_exp", "flash_lite", "lite", ""]
        for p in preferred:
            for flav in preferred_flavors:
                for m in models:
                    if p in m and (flav in m):
                        return m

        return models[0] if models else None
    except Exception:
        # Safe fallback — library commonly supports this
        return "gemini-1.5-flash"


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_multimodal_with_retry(prompt: str, img: Image.Image) -> Optional[str]:
    """Gemini image+text call with retry + key rotation."""
    global CURRENT_KEY
    if not HAS_GEMINI:
        return None

    model_name = resolve_multimodal_model_name()
    if not model_name:
        return None

    last_err = None
    payload_img = img
    # Some environments behave better with RGB
    try:
        payload_img = img.convert("RGB")
    except Exception:
        payload_img = img

    for _ in range(max(2, len(API_KEYS) * 2)):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content([prompt, payload_img])
            return getattr(resp, "text", None) or None
        except exceptions.ResourceExhausted as e:
            last_err = e
            if len(API_KEYS) > 1:
                CURRENT_KEY = next(KEY_CYCLE)
                genai.configure(api_key=CURRENT_KEY)
            time.sleep(0.6)
            continue
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "billing" in msg or "rate" in msg:
                return None
            time.sleep(0.2)
            continue

    _ = last_err
    return None


def _safe_json_loads(s: str) -> Optional[dict]:
    if not s:
        return None
    # Try direct JSON first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract JSON block
    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None



@st.cache_data(ttl=3600, show_spinner=False)
def gemini_palette_advice_cached(img_hash: str, context_prompt: str) -> Optional[dict]:
    """Lightweight Gemini text call to ensure palette_advice exists (fallback when vision JSON omits it)."""
    if not HAS_GEMINI:
        return None
    try:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(context_prompt)
        txt = getattr(resp, "text", None) or ""
        return _safe_json_loads(txt)
    except Exception:
        return None


def ensure_palette_advice(insight: dict, undertone: str, depth: str, label: str, pal: list, morning: list, evening: list) -> dict:
    """If Gemini vision output didn't include palette_advice, ask Gemini (text) to produce it using existing context."""
    if not insight:
        return insight
    padv = (insight.get("palette_advice") or {})
    has_any = any(padv.get(k) for k in ["morning_best", "evening_best", "avoid"])
    if has_any:
        return insight

    # Build a strict JSON-only prompt
    img_hash = st.session_state.get("img_hash", "img")
    prompt = f"""Return ONLY valid JSON. No markdown.

You are a personal stylist. Create palette guidance for a user based on complexion and the provided candidate colors.

Context:
- undertone={undertone}
- depth={depth}
- label={label}
- candidate_best_colors={pal[:10]}
- candidate_morning_colors={morning[:8]}
- candidate_evening_colors={evening[:8]}

Return JSON with:
- palette_advice: {{ "morning_best": [..1-6..], "evening_best": [..1-6..], "avoid": [..1-6..] }}
- color_pairings: 3 short strings like "Camel + Cream + Gold"
- explanation: 2-3 sentences why these colors work for this complexion
"""
    j = gemini_palette_advice_cached(img_hash, prompt) or {}
    if isinstance(j, dict):
        if j.get("palette_advice"):
            insight["palette_advice"] = j.get("palette_advice")
        if j.get("color_pairings"):
            insight["color_pairings"] = j.get("color_pairings")
        if j.get("explanation"):
            insight["explanation"] = j.get("explanation")
    return insight
@st.cache_data(ttl=3600, show_spinner=False)
def gemini_image_style_insight_cached(img_hash: str, prompt: str) -> Optional[dict]:
    """Cached wrapper so repeated reruns don't burn quota."""
    if not HAS_GEMINI:
        return None
    img = get_image_from_state()
    if img is None:
        return None
    txt = generate_multimodal_with_retry(prompt, img)
    return _safe_json_loads(txt) if txt else None


def gemini_image_style_insight(img: Image.Image, style: dict, country: str, category: str) -> Optional[dict]:
    """Returns structured Gemini insights from the upload image."""
    if not HAS_GEMINI:
        return None

    comp = (style.get("complexion") or {})
    undertone = comp.get("undertone") or style.get("undertone") or ""
    depth = comp.get("depth") or style.get("depth") or ""
    label = style.get("label") or ""
    vibe = style.get("style_vibe") or ""
    pal = [p.get("name") for p in (style.get("expert_palette") or []) if p.get("name")]
    morning = [p.get("name") for p in (style.get("morning_palette") or []) if p.get("name")]
    evening = [p.get("name") for p in (style.get("evening_palette") or []) if p.get("name")]

    # Keep prompt structured for reliability. We explicitly ask Gemini to *use* the complexion
    # and to output actionable color guidance (not just an image description).
    prompt = f"""You are a fashion stylist assistant for a Gemini 3 hackathon demo.
Analyze the photo and return STRICT JSON only (no markdown) with keys:
  - outfit_summary: short 1-2 sentence description of what you see (garments, vibe, formality)
  - formality: one of ["casual","smart casual","work","evening","formal"]
  - style_keywords: array of 6-10 style tags (e.g., "minimalist","streetwear","tailored","romantic")
  - IMPORTANT: clothing_items must have at least 3 items if any clothing is visible; infer missing basics if needed.
  - clothing_items: array of 3-6 clothing pieces you see (ONLY clothing; no jewelry/bags/shoes). Examples: "blazer","scoop-neck top","tailored trousers","midi skirt","saree","kurta set".
  - accessory_items: array of 2-6 accessories you see (jewelry/bag/shoes). Examples: "gold stud earrings","pendant necklace","tote bag".
  - garment_items: (legacy) array of key items you see.
  - recommended_shop_keywords: array of 6 shopping queries (no brand names) for the user's {category} collection in {country}. If you detect Indian wear, include 2 queries for Indian/ethnic outfits and accessories.
  - accessory_recos: object whose keys depend on {category}:
      * If category is Men: keys are watch, belt, sunglasses (optional bag as "briefcase" or "backpack"), and DO NOT include shoes/handbag/jewelry.
      * If category is Women: keys are shoes, handbag, jewelry.
    Each value is an array of 3 short suggestions.
  - is_indian_wear: boolean (true if the outfit looks like Indian/ethnic wear such as saree, salwar, kurta, lehenga)
  - indian_wear_type: string or empty (e.g., "saree", "kurta set", "lehenga")
  - indian_wear_addons: array of 4-8 add-ons appropriate for Indian wear (e.g., "jhumkas", "bangles", "bindi", "potli bag", "juttis", "dupatta")
  - palette_advice: object with keys morning_best, evening_best, avoid. Each value is an array of color names (1-6 items).
  - color_pairings: array of 3 short pairings like "Camel + Cream + Gold".
  - explanation: 2-3 short sentences explaining WHY these colors complement the user (based on complexion).

Context you MUST use (from local CV + palette engine):
  - User complexion profile: undertone={undertone}, depth={depth}, label={label}.
  - Candidate best colors: {pal[:10]}
  - Morning candidates: {morning[:8]}
  - Evening candidates: {evening[:8]}

If face isn't visible, still provide palette_advice using the candidate colors above.
"""

    # Cache by image hash + prompt template (style details are in prompt so keep stable)
    img_hash = st.session_state.get("img_hash", "img")
    return gemini_image_style_insight_cached(img_hash, prompt)
# -----------------------------
# LOOKBOOK + CONTEXT
# -----------------------------
COUNTRIES = {
    "United States": {"hl": "en-US", "gl": "US", "retailers": {"Nordstrom": "https://www.nordstrom.com/sr?keyword={q}", "Amazon": "https://www.amazon.com/s?k={q}"}},
    "India": {"hl": "en-IN", "gl": "IN", "retailers": {"Myntra": "https://www.myntra.com/search?q={q}", "Ajio": "https://www.ajio.com/search/?text={q}", "Amazon.in": "https://www.amazon.in/s?k={q}"}},
    "United Kingdom": {"hl": "en-GB", "gl": "GB", "retailers": {"ASOS": "https://www.asos.com/search/?q={q}", "Zara": "https://www.zara.com/uk/en/search?searchTerm={q}", "Amazon UK": "https://www.amazon.co.uk/s?k={q}"}},
    "France": {"hl": "fr-FR", "gl": "FR", "retailers": {"Galeries Lafayette": "https://www.galerieslafayette.com/s/{q}", "Zara": "https://www.zara.com/fr/en/search?searchTerm={q}", "Amazon.fr": "https://www.amazon.fr/s?k={q}"}},
    "Italy": {"hl": "it-IT", "gl": "IT", "retailers": {"Yoox": "https://www.yoox.com/search?query={q}", "Zara": "https://www.zara.com/it/en/search?searchTerm={q}", "Amazon.it": "https://www.amazon.it/s?k={q}"}},
    "Japan": {"hl": "ja-JP", "gl": "JP", "retailers": {"Rakuten Fashion": "https://search.rakuten.co.jp/search/mall/{q}/", "Amazon JP": "https://www.amazon.co.jp/s?k={q}"}},
    "South Korea": {"hl": "ko-KR", "gl": "KR", "retailers": {"MUSINSA": "https://www.musinsa.com/search/goods?keyword={q}", "Coupang": "https://www.coupang.com/np/search?component=&q={q}", "Naver Shopping": "https://search.shopping.naver.com/search/all?query={q}"}},
}

CELEB_DB = {
    "United States": {"Women": ["Zendaya", "Hailey Bieber", "Bella Hadid", "Rihanna", "Kendall Jenner", "Gigi Hadid"], "Men": ["Timothée Chalamet", "Jacob Elordi", "A$AP Rocky", "Ryan Gosling", "Harry Styles", "Donald Glover"]},
    "India": {"Women": ["Deepika Padukone", "Priyanka Chopra", "Alia Bhatt", "Sonam Kapoor", "Kiara Advani", "Janhvi Kapoor"], "Men": ["Ranveer Singh", "Vicky Kaushal", "Hrithik Roshan", "Ranbir Kapoor", "Shah Rukh Khan", "Ayushmann Khurrana"]},
    "United Kingdom": {"Women": ["Dua Lipa", "Florence Pugh", "Emma Watson", "Rosie Huntington-Whiteley", "Saoirse Ronan"], "Men": ["Harry Styles", "David Beckham", "Idris Elba", "Tom Hardy", "Robert Pattinson"]},
    "France": {"Women": ["Lily-Rose Depp", "Marion Cotillard", "Léa Seydoux", "Charlotte Gainsbourg", "Jeanne Damas"], "Men": ["Omar Sy", "Vincent Cassel", "Pierre Niney", "Jean Dujardin", "Tahar Rahim"]},
    "Italy": {"Women": ["Monica Bellucci", "Chiara Ferragni", "Bianca Balti", "Vittoria Ceretti"], "Men": ["Damiano David", "Michele Morrone", "Riccardo Scamarcio", "Gianluca Vacchi"]},
    "Japan": {"Women": ["Nana Komatsu", "Kiko Mizuhara", "Suzu Hirose", "Satomi Ishihara"], "Men": ["Kento Yamazaki", "Takuya Kimura", "Masaki Suda", "Ryoma Takeuchi"]},
    "South Korea": {"Women": ["Jennie (BLACKPINK)", "Han So-hee", "IU", "Song Hye-kyo"], "Men": ["G-Dragon", "Park Seo-joon", "V (BTS)", "Cha Eun-woo"]},
}


@st.cache_data(ttl=3600)
def get_country_context(country: str, gender: str):
    cfg = COUNTRIES.get(country, COUNTRIES["United States"])
    # Force English RSS (fixes Japan/KR showing non-English)
    rss_lang = "en"
    rss_url = (
        f"https://news.google.com/rss/search?q={_enc('fashion trend street style')}"
        f"&hl={rss_lang}&gl={cfg['gl']}&ceid={cfg['gl']}:{rss_lang}"
    )

    headlines = []
    try:
        feed = feedparser.parse(rss_url)
        headlines = list(dict.fromkeys([e.title for e in feed.entries]))[:7]
    except Exception:
        headlines = []

    if not headlines:
        headlines = ["Minimalist layering", "Vintage revival", "Oversized silhouettes"]

    local_celebs = (CELEB_DB.get(country, {}).get(gender) or CELEB_DB["United States"][gender])[:6]
    return headlines, local_celebs


def pills_html(items: List[str]) -> str:
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not items:
        return ""
    return "<div class='pill-row'>" + "".join([f"<span class='pill small'>{urllib.parse.quote_plus(i).replace('+',' ')}</span>" for i in items]) + "</div>"


def palette_chips_html(pal: List[dict], size_px: int = 38) -> str:
    if not pal:
        return ""
    parts = []
    for p in pal[:10]:
        cname = (str(p.get("name", "Color")).replace("'", "").strip() or "Color")
        chex = (str(p.get("hex", "#dddddd")).replace("'", "").strip() or "#DDDDDD")
        parts.append(
            f"<div class='chip'>"
            f"<div class='swatch' style='width:{size_px}px;height:{size_px}px;background:{chex};'></div>"
            f"<div class='chip-label'>{cname}</div>"
            f"<div class='chip-hex'>{chex}</div>"
            f"</div>"
        )
    return "<div class='chip-wrap'>" + "".join(parts) + "</div>"


@st.cache_data(ttl=86400)
def get_celebrity_image(name: str) -> str:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": name,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 900,
            "origin": "*",
            "redirects": 1,
        }
        headers = {"User-Agent": "LOOKBOOK-AI/1.0 (Streamlit)"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = page.get("thumbnail", {})
            if "source" in thumb:
                return thumb["source"]
    except Exception:
        pass
    return f"https://placehold.co/900x1100?text={_enc(name)}"


def build_celeb_google_images_url(celeb_name: str, style: dict, color: Optional[str] = None) -> str:
    """Google Images query.

    If a specific palette color is selected, bias the search strongly toward that color.
    Otherwise, include the top few palette colors as gentle hints.
    """
    label = str(style.get("label", "")).strip()
    base = [celeb_name, label]
    if color:
        base.append(color)
    else:
        colors = unique_keep_order([p.get("name", "") for p in (style.get("expert_palette") or [])])[:3]
        base.extend(colors)
    base += ["outfit", "street style"]
    q = " ".join([x for x in base if x]).strip()
    return f"https://www.google.com/search?tbm=isch&q={_enc(q)}"


def build_pinterest_url(celeb_name: str, style: dict, color: Optional[str] = None, category: Optional[str] = None) -> str:
    label = str(style.get("label", "")).strip()
    base = [celeb_name, label]
    # Add gender hint so Pinterest doesn't drift to women's results for men.
    if category:
        base.insert(0, "men" if str(category).strip().lower() == "men" else "women")
    if color:
        base.append(color)
    base += ["outfit", "street style"]
    q = " ".join([x for x in base if x]).strip()
    return f"https://www.pinterest.com/search/pins/?q={_enc(q)}"


STYLE_QUERY_HINTS: Dict[str, List[str]] = {
    # keep these short + retailer-friendly
    "Modern Minimal": ["minimal", "tailored"],
    "Quiet Luxury": ["tailored", "refined"],
    "Smart Casual": ["smart", "casual"],
}


def build_shop_query(
    category: str,
    keyword: str,
    style: dict,
    color: Optional[str] = None,
    retailer: Optional[str] = None,
) -> str:
    """Build a *short* query that works across retailers.

    Zara (and a few other retailers) often returns zero results for long, sentence-like queries.
    So we:
      - keep the query compact
      - use 1–2 style hint words instead of the whole style label
      - use 'man'/'woman' for Zara (their site taxonomy)
    """

    keyword = str(keyword or "").strip()
    label = str(style.get("label", "")).strip()
    retailer_l = (retailer or "").strip().lower()

    # gender token
    if retailer_l == "zara":
        gender_hint = "woman" if category == "Women" else "man"
    else:
        gender_hint = "women" if category == "Women" else "men"

    # style hint words
    hints = STYLE_QUERY_HINTS.get(label, [])

    parts: List[str] = []
    # Zara: lead with product keyword; gender + hints after (best match rate)
    if retailer_l == "zara":
        parts.append(keyword)
        parts.append(gender_hint)
        parts += hints[:1]
    else:
        parts.append(gender_hint)
        parts.append(keyword)
        parts += hints[:2]

    if color:
        parts.append(str(color).strip())

    # remove empties + de-dupe while preserving order
    cleaned = []
    seen = set()
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(p)

    return " ".join(cleaned).strip()


def lookbook_fallback(country: str, style: dict, headlines: List[str], celebs: List[str], category: str) -> dict:
    pal = style.get("expert_palette", []) or []
    colors = ", ".join(unique_keep_order([p.get("name", "") for p in pal])[:4]) or "your palette"
    label = style.get("label", "Signature").strip()

    celeb_styling = []
    for c in celebs[:5]:
        celeb_styling.append(
            {"name": c, "wearing": f"Lean into {label} with {colors}: refined base layers, structured outerwear, clean accessories."}
        )

    default_keywords = ["Blazer", "Trousers", "Loafers"] if category == "Men" else ["Blazer", "Wide-leg pants", "Heels"]

    return {
        "trend_summary": f"Across {country}, the mood is refined and wearable: {', '.join(headlines[:3])}.",
        "style_translation": [
            f"Anchor looks in {colors}.",
            "Prioritize structure (clean shoulders, straight hems).",
            "Keep accessories minimal; let texture do the work.",
        ],
        "outfit_idea": f"A {label} edit: crisp top + tailored bottom + a sleek layer in {colors}.",
        "shop_keywords": default_keywords,
        "celeb_styling": celeb_styling,
    }


def gemini_lookbook_text(country: str, category: str, style: dict, headlines: List[str], celebs: List[str]) -> dict:
    prompt = f"""
You are a Vogue editor. Write in English only.

Country: {country}
Collection: {category}
Style JSON: {json.dumps(style)}
Trend headlines: {headlines}
Celebrities: {celebs}

Return ONLY valid JSON (no markdown), schema:
{{
  "trend_summary": "string",
  "style_translation": ["string","string","string"],
  "outfit_idea": "string",
  "shop_keywords": ["string","string","string"],
  "celeb_styling": [
    {{"name":"string","wearing":"string"}},
    {{"name":"string","wearing":"string"}},
    {{"name":"string","wearing":"string"}},
    {{"name":"string","wearing":"string"}},
    {{"name":"string","wearing":"string"}}
  ]
}}

Constraints:
- Make celeb_styling match the user's palette + style label explicitly.
- Keep each "wearing" to 2 short sentences max.
"""
    txt = generate_text_with_retry(prompt)
    data = extract_json(txt or "")
    if not data:
        return lookbook_fallback(country, style, headlines, celebs, category)

    data.setdefault("trend_summary", "")
    data.setdefault("style_translation", [])
    data.setdefault("outfit_idea", "")
    data.setdefault("shop_keywords", [])
    data.setdefault("celeb_styling", [])
    return data


# -----------------------------
# UI SCREENS
# -----------------------------


# -----------------------------
# GEMINI INSIGHT RENDERING
# -----------------------------
def render_gemini_insight_full(insight: dict, analysis_mode_ui: str):
    """Full-width Gemini insight block shown below the image and spans the page."""
    if not insight:
        return

    outfit_summary = insight.get("outfit_summary") or ""
    formality = insight.get("formality") or ""
    style_keywords = insight.get("style_keywords") or []
    garment_items = insight.get("garment_items") or []
    clothing_items = insight.get("clothing_items") or []
    accessory_items = insight.get("accessory_items") or []
    # Normalize: keep clothing_items clothing-only; if missing, derive from garment_items and filter out accessories.
    if not clothing_items and garment_items:
        clothing_items = [str(x).strip() for x in garment_items if str(x).strip()]
    _acc_tokens = ["earring","earrings","necklace","jewelry","stud","ring","bracelet","pendant","bangle","jhumka",
                  "bag","handbag","purse","tote","clutch","shoe","shoes","heels","sneaker","sneakers","boot","boots","sandal","sandals","jutti"]
    def _is_accessory(txt: str) -> bool:
        t = (txt or "").lower()
        return any(tok in t for tok in _acc_tokens)
    clothing_items = [c for c in clothing_items if c and not _is_accessory(c)]
    if not accessory_items and garment_items:
        accessory_items = [str(x).strip() for x in garment_items if str(x).strip() and _is_accessory(str(x))]
    def _dedupe(seq):
        seen=set(); out=[]
        for x in seq:
            k=(x or "").lower()
            if k and k not in seen:
                seen.add(k); out.append(x)
        return out
    clothing_items = _dedupe(clothing_items)
    accessory_items = _dedupe(accessory_items)

    st.markdown("### ✨ Gemini 3 Insight")
    if outfit_summary:
        st.markdown(f"<div class='card'><div class='h3'>Outfit summary</div><div class='muted'>{outfit_summary}</div></div>", unsafe_allow_html=True)

    meta_bits = []
    if formality:
        meta_bits.append(f"<b>Formality:</b> {formality}")
    if clothing_items or garment_items:
        _ki = (clothing_items or garment_items)[:8]
        meta_bits.append(f"<b>Key items:</b> {', '.join(_ki)}")
    if meta_bits:
        st.markdown("<div class='muted'>" + " · ".join(meta_bits) + "</div>", unsafe_allow_html=True)

    if style_keywords:
        st.markdown("<div class='h3' style='margin-top:10px;'>Style tags</div>", unsafe_allow_html=True)
        pills = "".join([f"<span class='pill'>{k}</span>" for k in style_keywords[:12]])
        st.markdown(f"<div class='pillwrap'>{pills}</div>", unsafe_allow_html=True)

    # Outfit-colors mode: show accessory recommendations (no color guidance)
    if analysis_mode_ui == "Complexion (recommended)":
        acc = insight.get("accessory_recos") or {}
        shoes = acc.get("shoes") or []
        bag = acc.get("handbag") or []
        jew = acc.get("jewelry") or []
        is_ind = bool(insight.get("is_indian_wear"))
        ind_type = insight.get("indian_wear_type") or ""
        addons = insight.get("indian_wear_addons") or []

        
        st.markdown("<div class='h3' style='margin-top:14px;'>Complete the look</div>", unsafe_allow_html=True)

        category = st.session_state.get("category", "Women")
        is_men = (str(category).strip().lower() == "men")

        if is_men:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**👟 Shoes**")
                for s in shoes[:5]:
                    st.markdown(f"• {_decode_plus(s)}")
            with c2:
                st.markdown("**⌚ Watch**")
                for s in (acc.get("watch") or [])[:5]:
                    st.markdown(f"• {_decode_plus(s)}")
            with c3:
                st.markdown("**🧷 Belt**")
                for s in (acc.get("belt") or [])[:5]:
                    st.markdown(f"• {_decode_plus(s)}")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**👟 Shoes**")
                for s in shoes[:5]:
                    st.markdown(f"• {_decode_plus(s)}")
            with c2:
                st.markdown("**👜 Handbag**")
                for s in bag[:5]:
                    st.markdown(f"• {_decode_plus(s)}")
            with c3:
                st.markdown("**💍 Jewelry**")
                for s in jew[:5]:
                    st.markdown(f"• {_decode_plus(s)}")

        if is_ind:
            st.markdown("<div class='h3' style='margin-top:14px;'>Indian wear add-ons</div>", unsafe_allow_html=True)
            if ind_type:
                st.caption(f"Detected: {ind_type}")
            if addons:
                st.markdown("<div class='pillwrap'>" + "".join([f"<span class='pill'>{_decode_plus(a)}</span>" for a in addons[:10]]) + "</div>", unsafe_allow_html=True)
        return

    # Complexion mode: show palette guidance
    padv = insight.get("palette_advice") or {}
    pairings = insight.get("color_pairings") or []
    explanation = insight.get("explanation") or ""

    if any(padv.get(k) for k in ["morning_best", "evening_best", "avoid"]):
        st.markdown("<div class='h3' style='margin-top:14px;'>Gemini color guidance (based on your complexion)</div>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("🌤 **Morning best**")
            for c in padv.get("morning_best", [])[:6]:
                st.markdown(f"• {_decode_plus(c)}")
        with g2:
            st.markdown("🌙 **Evening best**")
            for c in padv.get("evening_best", [])[:6]:
                st.markdown(f"• {_decode_plus(c)}")
        with g3:
            st.markdown("⛔ **Avoid**")
            for c in padv.get("avoid", [])[:6]:
                st.markdown(f"• {_decode_plus(c)}")

    if pairings:
        st.markdown("<div class='h3' style='margin-top:12px;'>Suggested pairings</div>", unsafe_allow_html=True)
        pills = "".join([f"<span class='pill'>{_decode_plus(p)}</span>" for p in pairings[:6]])
        st.markdown(f"<div class='pillwrap'>{pills}</div>", unsafe_allow_html=True)

    if explanation:
        st.markdown("<div class='h3' style='margin-top:12px;'>Why these work</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>{explanation}</div>", unsafe_allow_html=True)



def render_center_upload_panel():
    """Centered upload step shown until an image is committed to session_state.

    UX goals:
    - Make it obvious where to upload
    - Avoid an awkward 'dead bottom' by explaining what happens next
    """
    st.markdown("## Step 1 — Upload a photo")
    st.caption("Best results: face visible, natural daylight, no heavy filters. JPG/PNG/WebP works great.")

    #st.info(
       #     "⬇️ Upload your image here to get your Gemini insights. "
       # "After you upload, the image controls move to the left sidebar so you can swap images anytime."
   # )

    uploaded = st.file_uploader(
        "Upload a selfie (face-visible works best for complexion)",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader_center",
        label_visibility="visible",
    )

    if uploaded is not None:
        # Preview stays compact and centered
        try:
            st.image(uploaded, caption="Preview", use_container_width=True)
        except Exception:
            pass

    c1, c2 = st.columns([1, 1])
    with c1:
        use = st.button(
            "✅ Use this upload",
            use_container_width=True,
            disabled=(uploaded is None),
            key="btn_use_center",
        )
    with c2:
        clear = st.button("🗑️ Clear", use_container_width=True, key="btn_clear_center")

    if clear:
        for k in ["img_bytes", "img_name", "img_mime", "img_hash", "style", "lookbook", "gemini_insight"]:
            st.session_state.pop(k, None)
        st.rerun()

    if use and uploaded is not None:
        save_uploaded_file_to_state(uploaded)
        # Reset derived state
        st.session_state.pop("style", None)
        st.session_state.pop("lookbook", None)
        st.session_state.pop("gemini_insight", None)
        st.rerun()

    # Make the bottom feel intentional instead of empty
    st.markdown(
        '''
        <div class="card" style="margin-top:16px;">
            <div style="font-weight:800; font-size:16px; margin-bottom:6px;">What happens next</div>
            <div class="small-muted" style="line-height:1.5;">
                <div>1) We analyze your complexion + outfit cues to recommend a palette.</div>
                <div>2) You'll get <b>Gemini Insights</b> and one-click access to <b>Lookbook & Shopping</b> curated to your colors.</div>
                <div style="margin-top:10px;">Tip: Use a front-facing photo in daylight. Avoid strong filters and colored lighting.</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )



def render_sidebar_image_controls():
    """Sidebar controls to swap/clear image after one has been uploaded."""
    st.markdown("### 📸 Image")
    uploaded = st.file_uploader(
        "Change image",
        type=["jpg", "png", "jpeg", "webp"],
        key="uploader_sidebar",
        label_visibility="visible",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Use", use_container_width=True, key="btn_use_sidebar"):
            if uploaded is not None:
                save_uploaded_file_to_state(uploaded)
                for k in ["style", "lookbook", "gemini_insight"]:
                    st.session_state.pop(k, None)
                st.rerun()
            else:
                st.warning("Upload an image first.")
    with colB:
        if st.button("🗑️ Clear", use_container_width=True, key="btn_clear_sidebar"):
            for k in ["img_bytes", "img_name", "img_mime", "img_hash", "style", "lookbook", "gemini_insight"]:
                st.session_state.pop(k, None)
            st.rerun()


def render_upload_screen():
    st.markdown(
        '''
        <div class="hero">
            <h1>LOOKBOOK AI</h1>
            <p>Global Style Intelligence · Personalized Curation</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    img = get_image_from_state()
    has_img = img is not None

    # Sidebar controls (preferences always; image controls move here after upload)
    with st.sidebar:
        st.markdown("### ⚙️ Preferences")

        category = st.radio(
            "Collection",
            ["Women", "Men"],
            index=(0 if st.session_state.get("category", "Women") == "Women" else 1),
            horizontal=True,
            key="sb_category_radio",
        )
        country = st.selectbox(
            "Region",
            list(COUNTRIES.keys()),
            index=list(COUNTRIES.keys()).index(st.session_state.get("country", "United States")),
            key="sb_country",
        )

        st.session_state["category"] = category
        st.session_state["country"] = country

        st.markdown("---")

        # Single-mode for hackathon: always use complexion
        st.session_state["analysis_mode_ui"] = "Complexion (recommended)"
        st.session_state["analysis_mode_committed"] = "Complexion (recommended)"

        st.markdown("---")

        if has_img:
            # After an image exists, image controls live in the sidebar so the user can swap anytime.
            render_sidebar_image_controls()
        else:
            st.markdown("### 📸 Image")
            st.caption("Upload in the center panel to begin. After you upload, controls appear here for quick changes.")

        st.markdown("---")
        if st.button("↻ Reset app", use_container_width=True, key="btn_reset_sidebar"):
            st.session_state.clear()
            st.rerun()

        st.caption("ℹ️ Palette is analyzed locally. If a Gemini API key is set, Gemini also analyzes your photo (vision) for outfit style + accessories, and generates the regional lookbook.")
# Top status bar (compact + always visible)
    category = st.session_state.get("category", "Women")
    country = st.session_state.get("country", "United States")
    mode = st.session_state.get("analysis_mode_ui", "Complexion (recommended)")
    st.markdown(
        f"<div class='card' style='padding:14px 16px; display:flex; justify-content:space-between; align-items:center;'>"
        f"<div><b>Selected</b> · {category} · {country}</div>"
        f"<div class='small-muted'>Analysis: {mode}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if not has_img:
        # Teach the user where to upload: centered Step 1 panel
        render_center_upload_panel()
        return


    col1, col2 = st.columns([1.05, 0.95], gap="large")

    with col1:
        st.markdown("<div class='sticky-left'>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Your Upload")
        st.caption(st.session_state.get("img_name", "uploaded image"))
        st.image(img, use_container_width=True)

        analyze_label = "✨ Analyze style" if "style" not in st.session_state else "🔁 Re-analyze (same image)"
        if st.button(analyze_label, type="primary", use_container_width=True):
            seed = st.session_state.get("img_hash", "seed")
            if st.session_state.get("analysis_mode_ui", "Complexion (recommended)") == "Complexion (recommended)":
                st.session_state["style"] = offline_style_profile(img, category, seed)
                st.session_state["style"]["analysis_mode"] = "Complexion (recommended)"
            else:
                st.session_state["style"] = offline_complexion_profile(img, category, seed)

            # Gemini multimodal (vision) — visible hackathon integration
            st.session_state.pop("gemini_insight", None)
            if HAS_GEMINI:
                with st.spinner("Gemini is analyzing your photo for style + accessories…"):
                    insight = gemini_image_style_insight(img, st.session_state["style"], country, category)
                if insight:
                    st.session_state["gemini_insight"] = insight
                    # Ensure palette guidance is present for complexion mode (Gemini vision may omit it)
                    try:
                        mode_ui = st.session_state.get("analysis_mode_ui", "Complexion")
                        if "complexion" in str(mode_ui).lower():
                            sty = st.session_state.get("style", {}) or {}
                            insight = ensure_palette_advice(
                                insight=insight,
                                undertone=str(sty.get("undertone", "")),
                                depth=str(sty.get("depth", "")),
                                label=str(sty.get("label", "")),
                                pal=list(sty.get("palette", []) or []),
                                morning=list(sty.get("morning_palette", []) or []),
                                evening=list(sty.get("evening_palette", []) or []),
                            )
                            st.session_state["gemini_insight"] = insight
                    except Exception:
                        pass

                    # Use Gemini style keywords as a stronger vibe hint for queries
                    if not st.session_state["style"].get("style_vibe") and insight.get("style_keywords"):
                        st.session_state["style"]["style_vibe"] = ", ".join(insight.get("style_keywords")[:4])
                    # If Gemini produced better shopping keywords, prefer them in lookbook
                    if insight.get("recommended_shop_keywords"):
                        st.session_state["style"]["gemini_shop_keywords"] = insight["recommended_shop_keywords"]

            st.session_state.pop("lookbook", None)
            st.rerun()


        st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.subheader("Photo tips (for best complexion results)")
        st.caption("Use a selfie/face-visible photo in daylight. Avoid heavy filters. If no face is detected, the app will fall back to outfit-based colors.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Gemini insight (moved to right panel)
        insight = st.session_state.get("gemini_insight")
        if insight:
            # First element inside the scroll panel: keep it flush to top
            st.markdown("<div class='card' style='margin-top:0px;'>", unsafe_allow_html=True)
            with st.expander("✨ Gemini 3 Insight (style + color reasoning)", expanded=True):
                if insight.get("outfit_summary"):
                    st.markdown(
                        "<div class='card card-tight' style='background:#f6f7fb;'>"
                        "<b>Outfit summary</b>"
                        f"<div class='small-muted' style='margin-top:6px;'>{insight.get('outfit_summary')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                meta = []
                if insight.get("formality"):
                    meta.append(f"Formality: <b>{insight.get('formality')}</b>")
                if insight.get("garment_items"):
                    meta.append("Key items: " + ", ".join([str(x) for x in insight.get("garment_items")[:6]]))
                if meta:
                    st.markdown("<div class='small-muted' style='margin-top:10px;'>" + " · ".join(meta) + "</div>", unsafe_allow_html=True)
                elif "complexion" in str(st.session_state.get("analysis_mode_ui","")).lower():
                    st.info("Gemini color guidance will appear after analysis. Click **Re-analyze** if you just switched modes.")

                if insight.get("style_keywords"):
                    kws = [str(x).strip() for x in insight.get("style_keywords") if str(x).strip()]
                    if kws:
                        st.markdown("<div style='margin-top:10px;'><b>Style tags</b></div>", unsafe_allow_html=True)
                        st.markdown(pills_html(kws[:12]), unsafe_allow_html=True)

                # Accessories (always useful; especially when user chose Complexion (recommended))

                # Accessories (Gemini)
                acc = insight.get("accessory_recos") or {}
                category_l = str(st.session_state.get("collection", "Women")).strip().lower()
                is_men = category_l.startswith("men")

                if isinstance(acc, dict):
                    if is_men:
                        # Men: show Watch/Belt/Sunglasses only (no shoes/handbag/jewelry)
                        if any(acc.get(k) for k in ["watch", "belt", "sunglasses"]):
                            st.markdown("<div style='margin-top:12px;'><b>Complete the look (Gemini)</b></div>", unsafe_allow_html=True)
                            a1, a2, a3 = st.columns(3, gap="medium")
                            with a1:
                                st.caption("⌚ Watch")
                                st.markdown(pills_html((acc.get("watch") or [])[:6]), unsafe_allow_html=True)
                            with a2:
                                st.caption("🧷 Belt")
                                st.markdown(pills_html((acc.get("belt") or [])[:6]), unsafe_allow_html=True)
                            with a3:
                                st.caption("🕶️ Sunglasses")
                                st.markdown(pills_html((acc.get("sunglasses") or [])[:6]), unsafe_allow_html=True)
                    else:
                        # Women: Shoes/Handbag/Jewelry
                        if any(acc.get(k) for k in ["shoes", "handbag", "jewelry"]):
                            st.markdown("<div style='margin-top:12px;'><b>Complete the look (Gemini)</b></div>", unsafe_allow_html=True)
                            a1, a2, a3 = st.columns(3, gap="medium")
                            with a1:
                                st.caption("👠 Shoes")
                                st.markdown(pills_html((acc.get("shoes") or [])[:6]), unsafe_allow_html=True)
                            with a2:
                                st.caption("👜 Handbag")
                                st.markdown(pills_html((acc.get("handbag") or [])[:6]), unsafe_allow_html=True)
                            with a3:
                                st.caption("💍 Jewelry")
                                st.markdown(pills_html((acc.get("jewelry") or [])[:6]), unsafe_allow_html=True)


                # Optional: Indian wear cues
                if insight.get("is_indian_wear"):
                    st.markdown("<div style='margin-top:10px;'><b>Indian wear add‑ons</b></div>", unsafe_allow_html=True)
                    extras = insight.get("indian_wear_addons") or []
                    if extras:
                        st.markdown(pills_html(extras[:10]), unsafe_allow_html=True)

                # Gemini color guidance (complexion-only)
                padv = insight.get("palette_advice") or {}

                if any(padv.get(k) for k in ["morning_best", "evening_best", "avoid"]):
                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    st.markdown("<b>Gemini color guidance (based on your complexion)</b>", unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3, gap="medium")
                    with g1:
                        st.caption("🌤 Morning best")
                        st.markdown(pills_html((padv.get("morning_best") or [])[:8]), unsafe_allow_html=True)
                    with g2:
                        st.caption("🌙 Evening best")
                        st.markdown(pills_html((padv.get("evening_best") or [])[:8]), unsafe_allow_html=True)
                    with g3:
                        st.caption("🚫 Avoid")
                        st.markdown(pills_html((padv.get("avoid") or [])[:8]), unsafe_allow_html=True)

                if insight.get("color_pairings"):
                    pairs = [urllib.parse.unquote_plus(str(x)).strip() for x in (insight.get("color_pairings") or []) if str(x).strip()]
                    if pairs:
                        st.markdown("<div style='margin-top:10px;'><b>Suggested pairings</b></div>", unsafe_allow_html=True)
                        st.markdown(pills_html(pairs[:8]), unsafe_allow_html=True)

                if insight.get("explanation"):
                    st.markdown("<div style='margin-top:10px;'><b>Why these colors work</b></div>", unsafe_allow_html=True)
                    st.caption(str(insight.get("explanation")))
            
        
        else:
            # Helpful placeholder so the right panel never looks empty
            st.markdown("<div class='card' style='margin-top:0px;'>", unsafe_allow_html=True)
            with st.expander("✨ Gemini 3 Insight (style + color reasoning)", expanded=True):
                st.markdown("**Run analysis to unlock insights.**")
                st.caption("Click **Analyze style** on the left to generate color reasoning, outfit summary, and pairings for your region.")
                st.markdown(
                    "<div class='card card-tight' style='background:#f6f7fb;'>"
                    "<b>What you'll get</b>"
                    "<ul style='margin:8px 0 0 18px;'>"
                    "<li>Best palette + morning/evening suggestions</li>"
                    "<li>Why the colors work for you</li>"
                    "<li>Suggested outfit pairings</li>"
                    "</ul>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        style = st.session_state.get("style")

        # Next step CTA lives in the main flow (not the sidebar)
        if style:
            st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.markdown("#### 🚀 Next step")
            st.caption("Open your regional lookbook and shopping picks based on your suggested colors + style.")
            if st.button("Open Lookbook & Shopping", use_container_width=True, key="btn_open_lookbook_main"):
                news, celebs = get_country_context(
                    st.session_state.get("country", "United States"),
                    st.session_state.get("category", "Women"),
                )
                st.session_state["lookbook"] = gemini_lookbook_text(
                    st.session_state.get("country", "United States"),
                    st.session_state.get("category", "Women"),
                    style,
                    news,
                    celebs,
                )
                navigate_to("lookbook")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Analyze a photo to unlock Lookbook & Shopping.")

    # Ensure morning/evening palettes exist for UI tabs
    if style and (not style.get("morning_palette") or not style.get("evening_palette")):
        comp = (style.get('complexion') or {})
        undertone = comp.get('undertone','')
        depth = comp.get('depth','')
        img_hash = st.session_state.get('img_hash','')
        m_pal, e_pal = split_morning_evening(style.get("expert_palette") or [], undertone=undertone, depth=depth, img_hash=img_hash)
        style.setdefault("morning_palette", m_pal)
        style.setdefault("evening_palette", e_pal)
        if not style:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Your Style Profile")
            st.caption("Click **Analyze style** to generate your personalized palette.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        pal = style.get("expert_palette", []) or []
        morning = style.get("morning_palette", []) or []
        evening = style.get("evening_palette", []) or []

        st.markdown("<div class='card card-tight'>", unsafe_allow_html=True)
        st.subheader("Your Style Profile")

        # Clear, compact summary
        label = (style.get("label","") or "").strip()
        if label:
            st.markdown(f"<span class='badge'>Result: {label}</span>", unsafe_allow_html=True)

        meta_bits = []
        if style.get("analysis_mode"):
            meta_bits.append(f"Mode: {style.get('analysis_mode')}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        if style.get("complexion_note"):
            st.warning(style.get("complexion_note"))

        # Palettes in tabs (less scrolling, clearer mental model)
        tab_best, tab_morn, tab_eve = st.tabs(["✨ Best colors", "🌤 Morning", "🌙 Evening"])

        with tab_best:
            if pal:
                st.markdown(palette_chips_html(pal, size_px=34), unsafe_allow_html=True)
            else:
                st.caption("No palette returned yet. Click Analyze.")

        with tab_morn:
            if morning:
                st.caption("Lighter / softer shades for daylight & work.")
                st.markdown(palette_chips_html(morning, size_px=30), unsafe_allow_html=True)
            else:
                st.caption("Morning palette will appear after analysis.")

        with tab_eve:
            if evening:
                st.caption("Deeper / richer shades for dinner & night.")
                st.markdown(palette_chips_html(evening, size_px=30), unsafe_allow_html=True)
            else:
                st.caption("Evening palette will appear after analysis.")





def render_lookbook_screen():
    country = st.session_state.get("country", "United States")
    category = st.session_state.get("category", "Women")
    lb = st.session_state.get("lookbook", {}) or {}
    style = st.session_state.get("style", {}) or {}

    # If the user jumps here without clicking "Build my lookbook", generate it automatically.
    if not lb:
        with st.spinner("Building your regional lookbook (Gemini 3)…"):
            news, celebs = get_country_context(country, category)
            st.session_state["lookbook"] = gemini_lookbook_text(country, category, style, news, celebs)
            lb = st.session_state.get("lookbook", {}) or {}
    pal = style.get("expert_palette", []) or []

    # DEDUPE COLORS HERE (FIXES YOUR SCREENSHOT)
    color_names = unique_keep_order([p.get("name", "").strip() for p in pal if p.get("name")])

    morning_names = unique_keep_order([p.get("name", "").strip() for p in (style.get("morning_palette", []) or []) if p.get("name")])
    evening_names = unique_keep_order([p.get("name", "").strip() for p in (style.get("evening_palette", []) or []) if p.get("name")])

    top = st.columns([1, 3, 2])
    with top[0]:
        if st.button("← Back to Studio"):
            navigate_to("upload")

    with top[2]:
        new_country = st.selectbox(
            "Region",
            list(COUNTRIES.keys()),
            index=list(COUNTRIES.keys()).index(country),
            key="lb_country",
        )
        new_category = st.selectbox(
            "Collection",
            ["Women", "Men"],
            index=(0 if category == "Women" else 1),
            key="lb_category",
        )

        if (new_country != country) or (new_category != category):
            st.session_state["country"] = new_country
            st.session_state["category"] = new_category
            news, celebs = get_country_context(new_country, new_category)
            st.session_state["lookbook"] = gemini_lookbook_text(new_country, new_category, style, news, celebs)
            st.rerun()

        if st.button("Regenerate lookbook", use_container_width=True):
            news, celebs = get_country_context(new_country, new_category)
            st.session_state["lookbook"] = gemini_lookbook_text(new_country, new_category, style, news, celebs)
            st.rerun()

    st.markdown(
        f"<h1 style='text-align:center; margin: 6px 0 18px 0;'>THE {country.upper()} EDIT</h1>",
        unsafe_allow_html=True,
    )

    if pal:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Your palette**")
        st.markdown(palette_chips_html(pal, size_px=34), unsafe_allow_html=True)
        st.caption("Use the Pinterest dropdown on each celebrity card to browse a specific palette color.")
        # Morning/Evening palettes (if available)
        morning = style.get("morning_palette", []) or []
        evening = style.get("evening_palette", []) or []
        if morning and evening:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Morning palette (softer)**")
            st.markdown(palette_chips_html(morning, size_px=30), unsafe_allow_html=True)
            st.markdown("**Evening palette (richer)**")
            st.markdown(palette_chips_html(evening, size_px=30), unsafe_allow_html=True)
            st.caption("Tip: Morning colors are great for daywear. Evening colors pop under low light and look sharper in photos.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Trend Intelligence", "Curated Shopping"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info(lb.get("trend_summary", ""))
        for tip in (lb.get("style_translation", []) or [])[:6]:
            st.write("•", tip)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.subheader("Outfit Idea")
        st.write(lb.get("outfit_idea", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # OPTION A (FIXED): dropdown + retailer buttons in SAME ROW
    # prevents giant full-width dropdown
    # -----------------------------

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        retailers = COUNTRIES[country]["retailers"]
        retailer_items = list(retailers.items())

        st.markdown("### Curated Shopping (Gemini + palette)")
        st.caption("Choose **Morning** or **Evening** — the entire shopping experience uses that palette (colors + accessories).")

        tod_choice = st.radio(
            "Shop for",
            ["🌤 Morning (day)", "🌙 Evening (night)"],
            horizontal=True,
            key=f"shop_tod_{country}_{category}",
        )
        is_evening = ("Evening" in tod_choice)
        pal_list = evening_names if is_evening else morning_names
        tod_label = "Evening" if is_evening else "Morning"
        tod_key = "evening" if is_evening else "morning"
        if not pal_list:
            st.info("No time-of-day palette found. Re-run analysis with a face-visible photo for best results.")
            pal_list = color_names[:]  # fallback

        # Prefer Gemini-detected CLOTHING items for per-piece shopping (outerwear/top/bottom/dress).
        insight = st.session_state.get("gemini_insight") or {}
        clothing_items_shop = []
        if isinstance(insight, dict):
            clothing_items_shop = insight.get("clothing_items") or []
            if not clothing_items_shop:
                # Back-compat: derive clothing from garment_items by filtering out accessories
                _gi = insight.get("garment_items") or []
                _acc_tokens = ["earring","earrings","necklace","jewelry","stud","ring","bracelet","pendant","bangle","jhumka",
                               "bag","handbag","purse","tote","clutch","shoe","shoes","heels","sneaker","sneakers","boot","boots","sandal","sandals","jutti"]
                def _is_acc(txt: str) -> bool:
                    t = (txt or "").lower()
                    return any(tok in t for tok in _acc_tokens)
                clothing_items_shop = [str(x).strip() for x in _gi if str(x).strip() and not _is_acc(str(x))]
        # De-dupe preserving order
        seen=set(); _tmp=[]
        for x in clothing_items_shop:
            k=x.lower()
            if k not in seen:
                seen.add(k); _tmp.append(x)
        clothing_items_shop = _tmp

        # Fall back to Gemini shop keywords / lookbook keywords if garments are unavailable.
        shop_keys = (style.get("gemini_shop_keywords") or lb.get("shop_keywords") or [])
        shop_keys = [str(x).strip() for x in shop_keys if str(x).strip()]
        if not shop_keys:
            shop_keys = ["tailored blazer", "white top", "straight trousers"]

        # Build the final per-item shopping list
        items_to_shop = []
        for g in clothing_items_shop:
            items_to_shop.append(str(g).strip())
        if not items_to_shop:
            items_to_shop = shop_keys
        # If Gemini returns too few clothing items, infer basics so the shopping list is complete.
        if len(items_to_shop) < 3:
            basics = ["top", "bottom", "outerwear"] if category.lower() == "women" else ["shirt", "trousers", "jacket"]
            for b in basics:
                if all(b.lower() not in str(x).lower() for x in items_to_shop):
                    items_to_shop.append(b)

        for idx, k in enumerate(items_to_shop[:8]):
            st.markdown(f"### {k}")

            widths = [2.2] + [1.0] * len(retailer_items)
            row = st.columns(widths, gap="small")

            picked = row[0].selectbox(
                f"{tod_label} palette colors",
                ["All palette colors"] + pal_list[:8],
                key=f"shop_color_{tod_key}_{country}_{category}_{idx}_{k}",
                label_visibility="collapsed",
            )
            picked_color = None if picked == "All palette colors" else picked

            for j, (rname, tpl) in enumerate(retailer_items, start=1):
                qtext = build_shop_query(category, k, style, color=picked_color, retailer=rname)
                url = tpl.format(q=_enc_retailer(rname, qtext))
                label = f"{rname} ↗" if not picked_color else f"{rname} · {picked_color} ↗"
                row[j].link_button(label, url, use_container_width=True)

            
            # Gender-aware accessories: Women -> shoes/handbag/jewelry, Men -> shoes/watch/belt
            category_l = str(category).strip().lower()
            is_men = category_l.startswith("men")
            exp_title = "✨ Complete the look (watch • belt • sunglasses)" if is_men else "✨ Complete the look (shoes • bag • jewelry)"

            with st.expander(exp_title, expanded=False):
                acc_cols = st.columns(3, gap="small")

                insight = st.session_state.get("gemini_insight") or {}
                acc = (insight.get("accessory_recos") or {}) if isinstance(insight, dict) else {}

                def build_accessory_query(item_type: str) -> str:
                    sug = None
                    if item_type == "shoes":
                        sug = (acc.get("shoes") or [None])[0]
                    elif item_type == "handbag":
                        sug = (acc.get("handbag") or [None])[0]
                    elif item_type == "jewelry":
                        sug = (acc.get("jewelry") or [None])[0]
                    elif item_type == "watch":
                        sug = (acc.get("watch") or [None])[0]
                    elif item_type == "belt":
                        sug = (acc.get("belt") or [None])[0]
                    elif item_type == "sunglasses":
                        sug = (acc.get("sunglasses") or [None])[0]

                    base = sug if sug else f"{k} {item_type}"
                    base = f"{base} {tod_label.lower()} outfit"
                    if picked_color:
                        base = f"{picked_color} {base}"
                    st_hint = (style.get("style_vibe") or style.get("style") or "").strip()
                    if st_hint:
                        base = f"{base} {st_hint}"
                    return base

                items = [("watch", "⌚"), ("belt", "🧷"), ("sunglasses", "🕶️")] if is_men else [("shoes", "👟"), ("handbag", "👜"), ("jewelry", "💍")]

                for col, (item_type, icon) in zip(acc_cols, items):
                    q = build_accessory_query(item_type)
                    g = f"https://www.google.com/search?tbm=isch&q={_enc(q)}"
                    p = build_pinterest(q, category=category)
                    col.markdown(f"**{icon} {item_type.title()}**")
                    col.link_button("Google Images ↗", g, use_container_width=True)
                    col.link_button("Pinterest ↗", p, use_container_width=True)

            st.markdown("---")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)


    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader(f"{category} icons in {country} (matched to your palette + style)")

    celebs = lb.get("celeb_styling", []) or []
    if not celebs:
        st.caption("No celebrity styling returned.")
        return

    show = celebs[:6]
    rows = [show[i : i + 3] for i in range(0, len(show), 3)]

    for row_idx, row in enumerate(rows):
        ccols = st.columns(3, gap="large")
        for i, celeb in enumerate(row):
            with ccols[i]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                name = celeb.get("name", "")
                st.markdown(f"### {name}")

                img_url = get_celebrity_image(name)
                st.image(img_url, use_container_width=True)

                st.caption(celeb.get("wearing", ""))

                # Actions (color first → then explore): dropdown applies to BOTH Google + Pinterest
                color_key = f"pin_color_{country}_{category}_{row_idx}_{i}_{name}"
                # default selection
                raw_selected = st.session_state.get(color_key, "All palette colors")
                if color_names:
                    raw_selected = st.selectbox(
                        "Choose a palette color",
                        ["All palette colors"] + color_names[:10],
                        key=color_key,
                        label_visibility="collapsed",
                    )
                selected_color = None if raw_selected == "All palette colors" else raw_selected

                a1, a2 = st.columns(2, gap="small")
                with a1:
                    st.link_button(
                        "Google Images ↗" if not selected_color else f"Google Images · {selected_color} ↗",
                        build_celeb_google_images_url(name, style, color=selected_color),
                        use_container_width=True,
                    )
                with a2:
                    st.link_button(
                        "Pinterest ↗" if not selected_color else f"Pinterest · {selected_color} ↗",
                        build_pinterest_url(name, style, color=selected_color, category=category),
                        use_container_width=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# MAIN ROUTER
# -----------------------------
if st.session_state.view == "upload":
    render_upload_screen()
elif st.session_state.view == "lookbook":
    render_lookbook_screen()
