# app.py (FULL UPDATED — SAFE IN-APP CURATED SHOPPING + GOOGLE SHOPPING FALLBACK)
# ✅ Keeps Google Shopping + Pinterest
# ✅ Removes the big gray placeholder images (only shows real images if you add them)
# ✅ If no curated product matches a filter → automatically shows Google Shopping + Pinterest
# ✅ Sidebar includes Change image
# ✅ Gemini insight + Lookbook + Curated Shopping tabs

import os
import re
import json
import time
import hashlib
import urllib.parse
import itertools
import io
from typing import Optional, List, Dict, Tuple

import streamlit as st
import feedparser
import requests
from PIL import Image
from dotenv import load_dotenv

# OpenCV optional (Cloud-safe)
try:
    import cv2
except Exception:
    cv2 = None

import numpy as np

# Gemini (Google Generative AI)
import google.generativeai as genai
from google.api_core import exceptions


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="LOOKBOOK AI | Global Style Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.secrets.get("GEMINI_API_KEY", "")
    or st.secrets.get("GOOGLE_API_KEY", "")
)
keys_env = os.getenv("GEMINI_API_KEYS") or st.secrets.get("GEMINI_API_KEYS", "")

API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()] if keys_env else []
if API_KEY and not API_KEYS:
    API_KEYS = [API_KEY]

HAS_GEMINI = bool(API_KEYS)
KEY_CYCLE = itertools.cycle(API_KEYS) if API_KEYS else None
CURRENT_KEY = next(KEY_CYCLE) if API_KEYS else None
if HAS_GEMINI:
    genai.configure(api_key=CURRENT_KEY)


# -----------------------------
# SAFE CURATED SHOPPING CATALOG (EDIT THIS LATER)
# -----------------------------
# NOTE: We keep placeholders in the catalog, BUT we DO NOT RENDER placeholder images.
# If you add real product images later, they will show.

def ph_img(text: str) -> str:
    return f"https://placehold.co/600x750/png?text={urllib.parse.quote_plus(text)}"

CATALOG: List[Dict[str, object]] = [
    # ---------- UNITED STATES / WOMEN ----------
    {"country":"United States","category":"Women","item_type":"Blazer","title":"Camel Tailored Blazer","price":"$80–$160","merchant":"Demo Retailer","colors":["Camel","Beige","Cream"],"buy_url":"https://example.com","image_url":ph_img("Camel+Tailored+Blazer")},
    {"country":"United States","category":"Women","item_type":"Blazer","title":"Black Double-Breasted Blazer","price":"$90–$180","merchant":"Demo Retailer","colors":["Black","Charcoal"],"buy_url":"https://example.com","image_url":ph_img("Black+Blazer")},
    {"country":"United States","category":"Women","item_type":"Wide-leg pants","title":"Ivory Wide-Leg Trousers","price":"$60–$140","merchant":"Demo Retailer","colors":["Ivory","Cream","White"],"buy_url":"https://example.com","image_url":ph_img("Ivory+Wide-Leg+Pants")},
    {"country":"United States","category":"Women","item_type":"Knit sweater","title":"Forest Green Knit","price":"$45–$120","merchant":"Demo Retailer","colors":["Forest","Olive"],"buy_url":"https://example.com","image_url":ph_img("Forest+Knit")},
    {"country":"United States","category":"Women","item_type":"Structured coat","title":"Navy Long Coat","price":"$120–$260","merchant":"Demo Retailer","colors":["Navy","Denim"],"buy_url":"https://example.com","image_url":ph_img("Navy+Coat")},
    {"country":"United States","category":"Women","item_type":"Heels","title":"Burgundy Heels","price":"$40–$110","merchant":"Demo Retailer","colors":["Burgundy","Rust"],"buy_url":"https://example.com","image_url":ph_img("Burgundy+Heels")},
    {"country":"United States","category":"Women","item_type":"Loafers","title":"Chocolate Loafers","price":"$60–$140","merchant":"Demo Retailer","colors":["Chocolate","Camel"],"buy_url":"https://example.com","image_url":ph_img("Chocolate+Loafers")},
    {"country":"United States","category":"Women","item_type":"Dress","title":"Teal Slip Dress","price":"$70–$160","merchant":"Demo Retailer","colors":["Teal"],"buy_url":"https://example.com","image_url":ph_img("Teal+Slip+Dress")},
    {"country":"United States","category":"Women","item_type":"Handbag","title":"Black Structured Bag","price":"$80–$220","merchant":"Demo Retailer","colors":["Black","Charcoal"],"buy_url":"https://example.com","image_url":ph_img("Black+Bag")},
    {"country":"United States","category":"Women","item_type":"Jewelry","title":"Gold Minimal Jewelry Set","price":"$25–$90","merchant":"Demo Retailer","colors":["Mustard","Cream"],"buy_url":"https://example.com","image_url":ph_img("Gold+Jewelry")},

    # ---------- UNITED STATES / MEN ----------
    {"country":"United States","category":"Men","item_type":"Blazer","title":"Charcoal Blazer","price":"$120–$260","merchant":"Demo Retailer","colors":["Charcoal","Black","Slate"],"buy_url":"https://example.com","image_url":ph_img("Charcoal+Blazer")},
    {"country":"United States","category":"Men","item_type":"Shirt","title":"White Oxford Shirt","price":"$40–$110","merchant":"Demo Retailer","colors":["White","Ivory","Cream"],"buy_url":"https://example.com","image_url":ph_img("White+Oxford+Shirt")},
    {"country":"United States","category":"Men","item_type":"Jeans","title":"Dark Denim Straight Jeans","price":"$55–$140","merchant":"Demo Retailer","colors":["Denim","Navy"],"buy_url":"https://example.com","image_url":ph_img("Dark+Denim+Jeans")},
    {"country":"United States","category":"Men","item_type":"Sneakers","title":"White Minimal Sneakers","price":"$60–$160","merchant":"Demo Retailer","colors":["White","Ivory"],"buy_url":"https://example.com","image_url":ph_img("White+Sneakers")},
    {"country":"United States","category":"Men","item_type":"Coat","title":"Camel Overcoat","price":"$140–$320","merchant":"Demo Retailer","colors":["Camel","Beige","Tan"],"buy_url":"https://example.com","image_url":ph_img("Camel+Overcoat")},
    {"country":"United States","category":"Men","item_type":"Watch","title":"Minimal Steel Watch","price":"$70–$220","merchant":"Demo Retailer","colors":["Gray","Slate","Charcoal"],"buy_url":"https://example.com","image_url":ph_img("Steel+Watch")},
    {"country":"United States","category":"Men","item_type":"Belt","title":"Chocolate Leather Belt","price":"$25–$70","merchant":"Demo Retailer","colors":["Chocolate","Camel"],"buy_url":"https://example.com","image_url":ph_img("Leather+Belt")},

    # ---------- INDIA / WOMEN ----------
    {"country":"India","category":"Women","item_type":"Kurta set","title":"Ivory Kurta Set","price":"₹1,499–₹2,999","merchant":"Demo Retailer","colors":["Ivory","Cream","White"],"buy_url":"https://example.com","image_url":ph_img("Ivory+Kurta+Set")},
    {"country":"India","category":"Women","item_type":"Saree","title":"Burgundy Saree","price":"₹1,999–₹4,999","merchant":"Demo Retailer","colors":["Burgundy","Rust"],"buy_url":"https://example.com","image_url":ph_img("Burgundy+Saree")},
    {"country":"India","category":"Women","item_type":"Heels","title":"Nude Heels","price":"₹899–₹2,499","merchant":"Demo Retailer","colors":["Beige","Tan","Cream"],"buy_url":"https://example.com","image_url":ph_img("Nude+Heels")},
    {"country":"India","category":"Women","item_type":"Handbag","title":"Black Sling Bag","price":"₹799–₹2,199","merchant":"Demo Retailer","colors":["Black","Charcoal"],"buy_url":"https://example.com","image_url":ph_img("Black+Sling+Bag")},
    {"country":"India","category":"Women","item_type":"Jewelry","title":"Gold Earrings","price":"₹499–₹1,999","merchant":"Demo Retailer","colors":["Mustard","Cream"],"buy_url":"https://example.com","image_url":ph_img("Gold+Earrings")},

    # ---------- INDIA / MEN ----------
    {"country":"India","category":"Men","item_type":"Kurta","title":"Navy Kurta","price":"₹999–₹2,499","merchant":"Demo Retailer","colors":["Navy","Denim"],"buy_url":"https://example.com","image_url":ph_img("Navy+Kurta")},
    {"country":"India","category":"Men","item_type":"Shirt","title":"Cream Linen Shirt","price":"₹899–₹2,299","merchant":"Demo Retailer","colors":["Cream","Ivory","Beige"],"buy_url":"https://example.com","image_url":ph_img("Cream+Linen+Shirt")},
    {"country":"India","category":"Men","item_type":"Shoes","title":"Brown Loafers","price":"₹1,499–₹3,499","merchant":"Demo Retailer","colors":["Chocolate","Camel","Tan"],"buy_url":"https://example.com","image_url":ph_img("Brown+Loafers")},
    {"country":"India","category":"Men","item_type":"Watch","title":"Minimal Watch (Leather Strap)","price":"₹1,299–₹3,999","merchant":"Demo Retailer","colors":["Chocolate","Charcoal"],"buy_url":"https://example.com","image_url":ph_img("Minimal+Watch")},
]

ITEM_NORMALIZE = {
    "Structured coat": "Structured coat",
    "Coat": "Coat",
    "Blazer": "Blazer",
    "Wide-leg pants": "Wide-leg pants",
    "Knit sweater": "Knit sweater",
    "Heels": "Heels",
    "Loafers": "Loafers",
    "Sneakers": "Sneakers",
    "Jeans": "Jeans",
    "Shirt": "Shirt",
    "Dress": "Dress",
    "Handbag": "Handbag",
    "Jewelry": "Jewelry",
    "Kurta": "Kurta",
    "Kurta set": "Kurta set",
    "Saree": "Saree",
    "Shoes": "Shoes",
    "Watch": "Watch",
    "Belt": "Belt",
}

def normalize_item_type(s: str) -> str:
    s = (s or "").strip()
    return ITEM_NORMALIZE.get(s, s)

def catalog_search(country: str, category: str, item_type: str, color: str, limit: int = 9) -> List[dict]:
    item_type = normalize_item_type(item_type)
    color_norm = (color or "").strip().lower()

    hits = []
    for p in CATALOG:
        if p.get("country") != country:
            continue
        if p.get("category") != category:
            continue
        if normalize_item_type(p.get("item_type", "")) != item_type:
            continue

        if not color_norm or color_norm == "all palette colors":
            hits.append(p)
        else:
            cols = [str(c).strip().lower() for c in (p.get("colors") or [])]
            if color_norm in cols:
                hits.append(p)

    if color_norm and color_norm != "all palette colors" and not hits:
        for p in CATALOG:
            if p.get("country") == country and p.get("category") == category and normalize_item_type(p.get("item_type","")) == item_type:
                hits.append(p)

    return hits[:limit]

def render_product_grid(products: List[dict], cols: int = 3):
    """
    ✅ UPDATED: Does NOT show placeholder images.
    Only shows an image if it's a real URL and not placehold.co
    """
    if not products:
        return

    grid = st.columns(cols)
    for i, p in enumerate(products):
        with grid[i % cols]:
            img = (p.get("image_url") or "").strip()
            # ❌ remove placeholder image rendering
            if img and ("placehold.co" not in img):
                st.image(img, use_container_width=True)

            st.markdown(f"**{p.get('title','')}**")
            meta = " · ".join([x for x in [p.get("price",""), p.get("merchant","")] if str(x).strip()])
            if meta:
                st.caption(meta)

            link = p.get("buy_url") or ""
            if link:
                st.link_button("Buy / View ↗", link, use_container_width=True)


# -----------------------------
# UI STYLES
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

html, body { background: #f7f7f9 !important; }
.stApp { background: #f7f7f9 !important; }

.hero {
    background: linear-gradient(180deg, #0b0b0f 0%, #12121a 100%);
    color: #fff;
    padding: 42px 30px;
    border-radius: 18px;
    margin-bottom: 18px;
    text-align: center;
    box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}
.hero h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0;
    font-size: 56px !important;
}
.hero p {
    color: rgba(255,255,255,0.82) !important;
    margin: 10px 0 0 0;
    font-size: 18px !important;
}

.card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 18px;
    background: #fff;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.small-muted { color: rgba(0,0,0,0.58) !important; font-size: 13px !important; }

[data-testid="stSidebar"] {
    background: #f2f3f6 !important;
    border-right: 1px solid rgba(0,0,0,0.08);
    height: 100vh !important;
    min-height: 100vh !important;
    position: sticky !important;
    top: 0 !important;
    align-self: flex-start;
}
[data-testid="stSidebar"] > div:first-child {
    max-height: 100vh !important;
    height: 100vh !important;
    overflow-y: auto !important;
    padding: 16px 14px 24px 14px !important;
    box-sizing: border-box;
}

.stButton>button, .stLinkButton>a {
    border-radius: 14px !important;
    padding: 0.65rem 0.9rem !important;
    font-weight: 700 !important;
}

.chip-wrap { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start; }
.chip { display:flex; flex-direction:column; align-items:center; width:96px; }
.swatch { border-radius:18px; border:1px solid rgba(0,0,0,0.10); box-shadow:0 4px 14px rgba(0,0,0,0.10); }
.chip-label { margin-top:8px; font-size:13px !important; font-weight:800; text-align:center; }
.chip-hex { margin-top:2px; font-size:12px !important; opacity:0.68; text-align:center; }

.pill-row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.pill {
    display:inline-block;
    padding:6px 10px;
    border-radius:999px;
    background:#f2f3f6;
    border:1px solid rgba(0,0,0,0.10);
    font-weight:800;
    font-size:13px !important;
}
.pill.small { padding:4px 8px; font-size:12px !important; opacity:0.92; }
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)


# -----------------------------
# ROUTING
# -----------------------------
if "view" not in st.session_state:
    st.session_state.view = "upload"

def navigate_to(view_name: str):
    st.session_state.view = view_name
    st.rerun()


# -----------------------------
# UTILS
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

def pills_html(items: List[str]) -> str:
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not items:
        return ""
    return "<div class='pill-row'>" + "".join([f"<span class='pill small'>{i}</span>" for i in items]) + "</div>"

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

def clear_analysis_outputs():
    for k in ["style", "lookbook", "gemini_insight", "gemini_last_error", "gemini_text_model_used", "gemini_vision_model_used"]:
        st.session_state.pop(k, None)

def safe_json_loads(text: str) -> Optional[dict]:
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


# -----------------------------
# LINKS
# -----------------------------
def build_celeb_google_images_url(query: str) -> str:
    return f"https://www.google.com/search?tbm=isch&q={_enc(query)}"

def build_pinterest_url(query: str) -> str:
    return f"https://www.pinterest.com/search/pins/?q={_enc(query)}"

def build_google_shop_url(query: str) -> str:
    return f"https://www.google.com/search?tbm=shop&q={_enc(query)}"

def build_shop_query(color: str, item: str, category: str, country: str) -> str:
    parts = [p for p in [color, item, category, country] if str(p).strip()]
    return " ".join(parts)


# -----------------------------
# COUNTRIES + RETAILERS
# -----------------------------
COUNTRIES = {
    "United States": {"gl": "US"},
    "India": {"gl": "IN"},
    "United Kingdom": {"gl": "GB"},
    "Japan": {"gl": "JP"},
}

CELEB_DB = {
    "United States": {"Women": ["Zendaya", "Hailey Bieber", "Rihanna"], "Men": ["Ryan Gosling", "Harry Styles", "Donald Glover"]},
    "India": {"Women": ["Deepika Padukone", "Priyanka Chopra", "Alia Bhatt"], "Men": ["Ranveer Singh", "Hrithik Roshan", "Shah Rukh Khan"]},
    "United Kingdom": {"Women": ["Dua Lipa", "Florence Pugh", "Emma Watson"], "Men": ["Idris Elba", "Tom Hardy", "David Beckham"]},
    "Japan": {"Women": ["Kiko Mizuhara", "Nana Komatsu", "Rola"], "Men": ["Takeru Satoh", "Sho Hirano", "Kentaro Sakaguchi"]},
}

@st.cache_data(ttl=3600)
def get_country_context(country: str, gender: str):
    cfg = COUNTRIES.get(country, COUNTRIES["United States"])
    gl = cfg["gl"]
    rss_url = f"https://news.google.com/rss/search?q={_enc('fashion street style trends')}&hl=en&gl={gl}&ceid={gl}:en"
    headlines = []
    try:
        feed = feedparser.parse(rss_url)
        headlines = list(dict.fromkeys([e.title for e in feed.entries]))[:7]
    except Exception:
        pass
    if not headlines:
        headlines = ["Minimal layering", "Vintage revival", "Tailored silhouettes"]
    celebs = (CELEB_DB.get(country, {}).get(gender) or CELEB_DB["United States"][gender])[:6]
    return headlines, celebs


# -----------------------------
# LOCAL PALETTE ANALYSIS
# -----------------------------
FASHION_SWATCHES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Black", (18, 18, 18)), ("Charcoal", (54, 54, 58)), ("Slate", (88, 96, 110)), ("Gray", (152, 156, 162)),
    ("Ivory", (242, 238, 228)), ("Cream", (232, 231, 220)), ("Beige", (216, 198, 168)), ("Tan", (196, 160, 118)),
    ("Camel", (193, 154, 107)), ("Chocolate", (82, 54, 42)), ("Navy", (20, 36, 74)), ("Denim", (55, 88, 132)),
    ("Olive", (96, 106, 58)), ("Forest", (24, 68, 44)), ("Burgundy", (92, 30, 44)), ("Rust", (168, 82, 52)),
    ("Mustard", (199, 164, 50)), ("Blush", (222, 170, 170)), ("Lavender", (160, 140, 190)), ("Teal", (34, 128, 122)),
    ("White", (248, 248, 248)),
]

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _nearest_fashion_name(rgb: Tuple[int, int, int]) -> str:
    def dist2(a, b): return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2
    return min(FASHION_SWATCHES, key=lambda sw: dist2(rgb, sw[1]))[0]

def _rgb_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [x/255.0 for x in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def _dominant_palette(img: Image.Image, k: int = 8) -> List[Dict[str, str]]:
    im = img.copy()
    im.thumbnail((420, 420))
    q = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=12)
    palette = q.getpalette()
    color_counts = q.getcolors() or []
    color_counts.sort(reverse=True, key=lambda x: x[0])

    picked, used_hex, used_name = [], set(), set()
    for _, idx in color_counts:
        rgb = (palette[idx*3+0], palette[idx*3+1], palette[idx*3+2])
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

def offline_complexion_profile(img: Image.Image) -> dict:
    face = detect_face_box(img)
    pal = _dominant_palette(img, k=8)
    if not face:
        return {
            "label":"Outfit palette",
            "expert_palette":pal[:5],
            "morning_palette":pal[:5],
            "evening_palette":pal[:5],
            "analysis_mode":"Outfit (fallback)",
            "complexion":{"undertone":"","depth":""},
            "complexion_note":"No face detected — using outfit colors instead."
        }

    scored = [(_rgb_luminance(tuple(int(p["hex"].lstrip("#")[i:i+2], 16) for i in (0,2,4))), p) for p in pal]
    scored.sort(key=lambda t: t[0], reverse=True)
    morning = [p for _, p in scored[:5]]
    evening = [p for _, p in scored[-5:]][::-1]

    return {
        "label": "Complexion analysis",
        "expert_palette": pal[:5],
        "morning_palette": morning[:5],
        "evening_palette": evening[:5],
        "analysis_mode": "Complexion (recommended)",
        "complexion": {"undertone": "", "depth": ""},
    }


# -----------------------------
# GEMINI
# -----------------------------
def rotate_key():
    global CURRENT_KEY
    if KEY_CYCLE and len(API_KEYS) > 1:
        CURRENT_KEY = next(KEY_CYCLE)
        genai.configure(api_key=CURRENT_KEY)

def resolve_text_model_name() -> Optional[str]:
    if not HAS_GEMINI:
        return None
    override = os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL")
    if override:
        return override
    return "gemini-3-flash-preview"

def resolve_multimodal_model_name() -> Optional[str]:
    if not HAS_GEMINI:
        return None
    override = os.getenv("GEMINI_VISION_MODEL") or os.getenv("GEMINI_MODEL")
    if override:
        return override
    return "gemini-3-flash-preview"

def generate_text_with_retry(prompt: str) -> Optional[str]:
    if not HAS_GEMINI:
        return None
    model_name = resolve_text_model_name()
    last_err = None
    for _ in range(max(2, len(API_KEYS) * 2)):
        try:
            model = genai.GenerativeModel(model_name)
            st.session_state["gemini_text_model_used"] = model_name
            resp = model.generate_content(prompt)
            return getattr(resp, "text", None) or None
        except exceptions.ResourceExhausted as e:
            last_err = e
            rotate_key()
            time.sleep(0.6)
        except Exception as e:
            last_err = e
            time.sleep(0.25)
    st.session_state["gemini_last_error"] = f"{type(last_err).__name__}: {last_err}" if last_err else "Unknown error"
    return None

def generate_multimodal_with_retry(prompt: str, img: Image.Image) -> Optional[str]:
    if not HAS_GEMINI:
        return None
    model_name = resolve_multimodal_model_name()
    last_err = None
    for _ in range(max(2, len(API_KEYS) * 2)):
        try:
            model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
            st.session_state["gemini_vision_model_used"] = model_name
            resp = model.generate_content([prompt, img])
            return getattr(resp, "text", None) or None
        except exceptions.ResourceExhausted as e:
            last_err = e
            rotate_key()
            time.sleep(0.6)
        except Exception as e:
            last_err = e
            time.sleep(0.25)
    st.session_state["gemini_last_error"] = f"{type(last_err).__name__}: {last_err}" if last_err else "Unknown error"
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def gemini_image_style_insight_cached(img_hash: str, prompt: str, img_bytes: bytes) -> Optional[dict]:
    if not HAS_GEMINI or not img_bytes:
        return None
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    txt = generate_multimodal_with_retry(prompt, img)
    return safe_json_loads(txt or "")

def gemini_image_style_insight(style: dict, country: str, category: str) -> Optional[dict]:
    if not HAS_GEMINI:
        return None
    prompt = """
Return STRICT JSON only (no markdown).
Keys:
- outfit_summary (string)
- style_keywords (array)
- palette_advice { morning_best:[], evening_best:[], avoid:[] }
- color_pairings (array)
- explanation (string)
"""
    img_hash = st.session_state.get("img_hash", "img")
    img_bytes = st.session_state.get("img_bytes", b"")
    return gemini_image_style_insight_cached(img_hash, prompt, img_bytes)

def gemini_lookbook_text(country: str, category: str, style: dict, news: List[str], celebs: List[str]) -> dict:
    prompt = f"""
Return ONLY valid JSON (no markdown).

Schema:
{{
  "trend_summary": "string",
  "style_translation": ["string","string","string"],
  "outfit_idea": "string",
  "shop_keywords": ["string","string","string","string","string","string"]
}}

Country: {country}
Collection: {category}
Trend headlines: {news}
Celebrities: {celebs}
Palette: {style.get("expert_palette", [])}
"""
    txt = generate_text_with_retry(prompt) or ""
    j = safe_json_loads(txt)
    if not j:
        return {
            "trend_summary": f"Across {country}, the mood is refined and wearable: {', '.join(news[:3])}.",
            "style_translation": ["Anchor looks in your palette.", "Prioritize structure (clean shoulders, straight hems).", "Keep accessories minimal; let texture do the work."],
            "outfit_idea": "A modern minimal edit: crisp top + tailored bottom + a sleek layer in your palette.",
            "shop_keywords": ["Blazer", "Wide-leg pants", "Knit sweater", "Structured coat", "Loafers", "Sneakers"],
        }
    return j


# -----------------------------
# SIDEBAR IMAGE CONTROLS
# -----------------------------
def render_sidebar_image_controls():
    st.markdown("### 🖼️ Image")
    new_file = st.file_uploader("Change image", type=["jpg", "jpeg", "png", "webp"], key="sidebar_image_uploader")
    c1, c2 = st.columns(2)
    with c1:
        use_new = st.button("✅ Use", use_container_width=True, disabled=(new_file is None), key="sb_use_img")
    with c2:
        clear_img = st.button("🗑️ Clear", use_container_width=True, key="sb_clear_img")

    if clear_img:
        for k in ["img_bytes", "img_name", "img_hash"]:
            st.session_state.pop(k, None)
        clear_analysis_outputs()
        st.rerun()

    if use_new and new_file is not None:
        save_uploaded_file_to_state(new_file)
        clear_analysis_outputs()
        st.rerun()


# -----------------------------
# UPLOAD VIEW
# -----------------------------
def render_center_upload_panel():
    st.markdown("## Upload a photo")
    st.caption("Best results: face visible, natural daylight. JPG/PNG/WebP.")
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"], key="uploader_center")

    if uploaded is not None:
        st.image(uploaded, caption="Preview", use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        use = st.button("✅ Use this upload", use_container_width=True, disabled=(uploaded is None), key="center_use")
    with c2:
        clear = st.button("🗑️ Clear", use_container_width=True, key="center_clear")

    if clear:
        for k in ["img_bytes", "img_name", "img_hash"]:
            st.session_state.pop(k, None)
        clear_analysis_outputs()
        st.rerun()

    if use and uploaded is not None:
        save_uploaded_file_to_state(uploaded)
        clear_analysis_outputs()
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

    with st.sidebar:
        st.markdown("### Preferences")
        category = st.radio("Collection", ["Women", "Men"], horizontal=True, key="sb_category")
        country = st.selectbox("Region", list(COUNTRIES.keys()), key="sb_country")
        st.session_state["category"] = category
        st.session_state["country"] = country

        st.markdown("---")
        if has_img:
            render_sidebar_image_controls()
        else:
            st.caption("Upload an image to enable Change Image controls here.")

        st.markdown("---")
        used_v = st.session_state.get("gemini_vision_model_used")
        used_t = st.session_state.get("gemini_text_model_used")
        if HAS_GEMINI and (used_v or used_t):
            st.success(f"Gemini model used: {used_v or used_t}")
        elif HAS_GEMINI:
            st.info("Gemini: connected")
        else:
            st.warning("Gemini: not connected (missing API key)")

        if st.button("🧹 Clear cache", use_container_width=True, key="sb_clear_cache"):
            st.cache_data.clear()
            clear_analysis_outputs()
            st.rerun()

    if not has_img:
        render_center_upload_panel()
        return

    st.markdown(
        f"<div class='card' style='padding:14px 16px; display:flex; justify-content:space-between; align-items:center;'>"
        f"<div><b>Selected</b> · {category} · {country}</div>"
        f"<div class='small-muted'>Analysis: Complexion (recommended)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.08, 0.92], gap="large")

    def run_analysis():
        st.session_state["style"] = offline_complexion_profile(img)
        for k in ["lookbook", "gemini_insight", "gemini_last_error", "gemini_text_model_used", "gemini_vision_model_used"]:
            st.session_state.pop(k, None)

        if HAS_GEMINI:
            with st.spinner("Gemini is analyzing your photo…"):
                insight = gemini_image_style_insight(st.session_state["style"], country, category)
            if insight:
                st.session_state["gemini_insight"] = insight
        st.rerun()

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Your Upload")
        st.caption(st.session_state.get("img_name", "uploaded image"))
        st.image(img, use_container_width=True)
        if st.button("✨ Analyze style", type="primary", use_container_width=True, key="analyze_left"):
            run_analysis()
        st.markdown("</div>", unsafe_allow_html=True)

        style = st.session_state.get("style")
        if style:
            st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.subheader("Your palette")
            st.markdown(palette_chips_html(style.get("expert_palette", [])), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.write("Morning palette")
            st.markdown(palette_chips_html(style.get("morning_palette", []), size_px=30), unsafe_allow_html=True)
            st.write("Evening palette")
            st.markdown(palette_chips_html(style.get("evening_palette", []), size_px=30), unsafe_allow_html=True)
            if style.get("complexion_note"):
                st.caption(style["complexion_note"])
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.button("✨ Analyze style", type="primary", use_container_width=True, key="analyze_right"):
            run_analysis()

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
            st.caption("Generate a regional lookbook + curated shopping (Gemini text).")
            if st.button("Open Lookbook", use_container_width=True, key="open_lookbook"):
                news, celebs = get_country_context(country, category)
                with st.spinner("Building your regional lookbook…"):
                    lb = gemini_lookbook_text(country, category, style, news, celebs)
                st.session_state["lookbook"] = lb
                st.session_state["celebs_for_lookbook"] = celebs
                navigate_to("lookbook")
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# LOOKBOOK VIEW
# -----------------------------
def render_lookbook_screen():
    country = st.session_state.get("country", "United States")
    category = st.session_state.get("category", "Women")
    style = st.session_state.get("style", {}) or {}
    lb = st.session_state.get("lookbook", {}) or {}

    if not lb:
        news, celebs = get_country_context(country, category)
        with st.spinner("Building your regional lookbook…"):
            lb = gemini_lookbook_text(country, category, style, news, celebs)
        st.session_state["lookbook"] = lb
        st.session_state["celebs_for_lookbook"] = celebs

    celebs = st.session_state.get("celebs_for_lookbook") or (CELEB_DB.get(country, {}).get(category) or [])

    if st.button("← Back", key="back_to_upload"):
        navigate_to("upload")

    st.markdown(f"<h2 style='text-align:left; margin-top:6px;'>THE {country.upper()} EDIT</h2>", unsafe_allow_html=True)

    if style.get("expert_palette"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Your palette**")
        st.markdown(palette_chips_html(style.get("expert_palette", []), size_px=30), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Trend Intelligence", "Curated Shopping"])

    # -------------------- Trend Intelligence
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info(lb.get("trend_summary", ""))
        for tip in (lb.get("style_translation", []) or [])[:6]:
            st.write("·", tip)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.subheader("Outfit Idea")
        st.write(lb.get("outfit_idea", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.subheader(f"{category} icons in {country} (palette-matched)")

        palette_names = [p.get("name") for p in (style.get("expert_palette") or []) if p.get("name")] or []
        palette_hint = ", ".join(palette_names[:5]) if palette_names else "your palette"

        pin_color = st.selectbox("Pinterest color filter", ["All palette colors"] + palette_names, key="celeb_pin_color_filter")

        cols = st.columns(3)
        for idx, celeb in enumerate(celebs[:9]):
            with cols[idx % 3]:
                st.markdown(f"### {celeb}")
                st.caption(
                    f"Lean into your look with {palette_hint}: refined base layers, structured outerwear, clean accessories."
                )
                st.link_button("Google Images", build_celeb_google_images_url(f"{celeb} {country} street style"), use_container_width=True)

                pin_q = f"{celeb} {country} street style"
                if pin_color and pin_color != "All palette colors":
                    pin_q = f"{celeb} {pin_color} outfit"
                st.link_button("Pinterest", build_pinterest_url(pin_q), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Curated Shopping (IN-APP + GOOGLE SHOPPING FALLBACK)
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Curated Shopping")

        day_or_night = st.radio("Shop for", ["☀️ Morning (day)", "🌙 Evening (night)"], horizontal=True, key="shop_day_night")
        is_morning = day_or_night.startswith("☀️")

        palette = style.get("morning_palette") if is_morning else style.get("evening_palette")
        palette = palette or style.get("expert_palette") or []
        palette_names = unique_keep_order([p.get("name") for p in palette if p.get("name")])
        palette_dropdown = ["All palette colors"] + palette_names

        items = unique_keep_order((lb.get("shop_keywords") or [])[:10])
        if not items:
            items = ["Blazer", "Wide-leg pants", "Knit sweater", "Structured coat", "Loafers", "Sneakers"]

        accessories = ["Handbag", "Jewelry", "Heels"] if category == "Women" else ["Watch", "Belt", "Shoes"]

        for item in items:
            item_type = normalize_item_type(item)
            st.markdown(f"### {item_type}")

            chosen_color = st.selectbox(
                "Filter by your palette",
                palette_dropdown,
                key=f"catalog_color_{item_type}_{'m' if is_morning else 'e'}",
            )
            color_for_filter = "" if chosen_color == "All palette colors" else chosen_color

            products = catalog_search(country, category, item_type, color_for_filter, limit=9)

            st.markdown("#### Suggested picks")

            # ✅ UPDATED: If no curated products, show Google Shopping + Pinterest (and NO placeholder images)
            if products:
                render_product_grid(products, cols=3)
            else:
                st.info("No curated products found for this filter yet — showing Google Shopping instead.")
                q = build_shop_query(color_for_filter, item_type, category, country)
                st.link_button("Google Shopping ↗", build_google_shop_url(q), use_container_width=True)
                st.link_button("Pinterest ↗", build_pinterest_url(q), use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("### ✨ Complete the look")
        st.caption("Accessories (if missing, we show Google Shopping + Pinterest).")

        acc_cols = st.columns(3)
        for i, acc in enumerate(accessories):
            with acc_cols[i % 3]:
                st.markdown(f"**{acc}**")
                chosen_color = st.selectbox(
                    "Palette filter",
                    palette_dropdown,
                    key=f"acc_color_{acc}_{'m' if is_morning else 'e'}",
                    label_visibility="collapsed",
                )
                color_for_filter = "" if chosen_color == "All palette colors" else chosen_color
                products = catalog_search(country, category, acc, color_for_filter, limit=3)

                if products:
                    render_product_grid(products, cols=1)
                else:
                    q = build_shop_query(color_for_filter, acc, category, country)
                    st.link_button("Google Shopping ↗", build_google_shop_url(q), use_container_width=True)
                    st.link_button("Pinterest ↗", build_pinterest_url(q), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# MAIN
# -----------------------------
if st.session_state.view == "upload":
    render_upload_screen()
elif st.session_state.view == "lookbook":
    render_lookbook_screen()
