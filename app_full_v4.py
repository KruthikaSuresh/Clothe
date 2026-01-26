import os
import re
import json
import hashlib
import random
import urllib.parse
import time
import itertools
import requests

import streamlit as st
import feedparser
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions

# --- CONFIGURATION ---
st.set_page_config(
    page_title="LOOKBOOK AI | Global Style Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
keys_env = os.getenv("GEMINI_API_KEYS")
API_KEYS = [k.strip() for k in keys_env.split(',') if k.strip()] if keys_env else []
if API_KEY and not API_KEYS:
    API_KEYS = [API_KEY]

if not API_KEYS:
    st.error("⚠️ Missing API key. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

KEY_CYCLE = itertools.cycle(API_KEYS)
CURRENT_KEY = next(KEY_CYCLE)
genai.configure(api_key=CURRENT_KEY)

# --- STATE MANAGEMENT ---
if 'view' not in st.session_state:
    st.session_state.view = 'upload'

def navigate_to(view_name):
    st.session_state.view = view_name
    st.rerun()

# --- STYLING (Large Fonts & HTML Buttons) ---
STYLING = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Montserrat:wght@400;500;600&display=swap');
    
    /* GLOBAL FONTS - INCREASED SIZE */
    html, body, [class*="css"], .stMarkdown, p, li, div { 
        font-family: 'Montserrat', sans-serif !important;
        font-size: 20px !important;
        color: #1a1a1a;
        line-height: 1.6 !important;
    }
    
    h1, h2, h3, .hero-title, .section-header { 
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 700 !important;
        color: #000 !important;
    }
    
    /* HTML LINK BUTTON (Fixes Localhost Issue) */
    .custom-btn {
        display: inline-block;
        background-color: #000;
        color: #fff !important;
        padding: 12px 24px;
        text-decoration: none !important;
        border-radius: 4px;
        font-weight: 600;
        text-align: center;
        transition: all 0.2s;
        border: 1px solid #000;
        width: 100%;
        margin-top: 10px;
        font-size: 16px !important;
    }
    .custom-btn:hover {
        background-color: #333;
        transform: translateY(-2px);
    }

    .custom-btn-ghost {
        background-color: #fff;
        color: #000 !important;
        border: 2px solid #000;
    }
    .custom-btn-ghost:hover {
        background-color: #f0f0f0;
    }

    /* HERO */
    .hero-container {
        background: #000;
        color: white;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 40px;
    }
    .hero-title { font-size: 4rem !important; color: white !important; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.2rem !important; color: #ccc !important; }
    
    /* CARDS */
    .st-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    
    /* CELEB CARD (Name -> Image -> Desc) */
    .celeb-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        overflow: hidden;
        background: white;
        height: 100%;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .celeb-header {
        padding: 20px;
        text-align: center;
        background: #fff;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .celeb-name { 
        font-family: 'Cormorant Garamond', serif; 
        font-size: 2rem; 
        font-weight: 700; 
        margin: 0;
        color: #000;
    }

    .celeb-img-box {
        height: 400px;
        background: #f9f9f9;
        overflow: hidden;
        position: relative;
    }
    .celeb-img-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: top;
    }
    
    .celeb-info { 
        padding: 25px; 
        text-align: center; 
        flex-grow: 1; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .celeb-desc {
        font-size: 1.1rem;
        color: #444;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* HIDE STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] { min-width: 350px; }
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_valid_generative_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if "gemini-1.5-flash" in m: return m
        return models[0] if models else "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

ACTIVE_MODEL = get_valid_generative_model()

def generate_with_retry(model, content):
    global CURRENT_KEY
    try:
        return model.generate_content(content)
    except exceptions.ResourceExhausted:
        if len(API_KEYS) > 1:
            CURRENT_KEY = next(KEY_CYCLE)
            genai.configure(api_key=CURRENT_KEY)
            time.sleep(1)
            return model.generate_content(content)
        else:
            st.error("⏳ Limit reached. Please wait 30s...")
            st.stop()
    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

def extract_json(text: str):
    try:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n\t")
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except: pass
    return {}

def _enc(s: str) -> str:
    return urllib.parse.quote_plus(str(s).strip())

@st.cache_data(ttl=86400)
def get_celebrity_image(name):
    """Fetches a high-quality image from Wikipedia API."""
    try:
        url = "[https://en.wikipedia.org/w/api.php](https://en.wikipedia.org/w/api.php)"
        params = {
            "action": "query", "titles": name, "prop": "pageimages",
            "format": "json", "pithumbsize": 600, "origin": "*"
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "thumbnail" in page:
                return page["thumbnail"]["source"]
    except: pass
    return f"[https://placehold.co/600x800/eeeeee/333333?text=](https://placehold.co/600x800/eeeeee/333333?text=){_enc(name)}"

# --- CORE LOGIC ---

def gemini_style_profile(img, category, seed):
    model = genai.GenerativeModel(ACTIVE_MODEL)
    prompt = f"Professional stylist analysis for {category}. Output VALID JSON: {{'label':'Vibe','ideal_cut':'Fit','expert_palette':[{{'name':'Color','hex':'#HEX'}}],'items':['Item1']}}. Seed: {seed}"
    resp = generate_with_retry(model, [prompt, img])
    return extract_json(resp.text)

@st.cache_data(ttl=3600)
def get_country_context(country, gender):
    # RSS Trends
    cfg = COUNTRIES.get(country, COUNTRIES["United States"])
    rss_url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){_enc('fashion trend street style')}&hl={cfg['hl']}&gl={cfg['gl']}&ceid={cfg['gl']}:{cfg['hl']}"
    
    headlines = []
    try:
        feed = feedparser.parse(rss_url)
        headlines = list(set([e.title for e in feed.entries]))[:5]
    except: headlines = []
    if not headlines: headlines = ["Minimalist layering", "Vintage revival", "Oversized silhouettes"]

    # Celebrity DB
    celeb_db = {
        "United States": {
            "Women": ["Zendaya", "Hailey Bieber", "Bella Hadid"],
            "Men": ["Timothée Chalamet", "Jacob Elordi", "A$AP Rocky"]
        },
        "India": {
            "Women": ["Deepika Padukone", "Priyanka Chopra", "Alia Bhatt"],
            "Men": ["Ranveer Singh", "Vijay Varma", "Diljit Dosanjh"]
        },
        "United Kingdom": {
            "Women": ["Dua Lipa", "Florence Pugh", "Kate Moss"],
            "Men": ["Harry Styles", "David Beckham", "Idris Elba"]
        },
        "France": {
            "Women": ["Lily-Rose Depp", "Jeanne Damas", "Camille Rowe"],
            "Men": ["Lucas Bravo", "Timothée Chalamet", "Vincent Cassel"]
        },
        "Italy": {
            "Women": ["Chiara Ferragni", "Monica Bellucci", "Vittoria Ceretti"],
            "Men": ["Damiano David", "Mariano Di Vaio", "Michele Morrone"]
        },
        "Japan": {
            "Women": ["Nana Komatsu", "Kiko Mizuhara", "Rola"],
            "Men": ["Kento Yamazaki", "Takuya Kimura", "Verdy"]
        },
        "South Korea": {
            "Women": ["Jennie (Blackpink)", "Han So-hee", "Song Hye-kyo"],
            "Men": ["G-Dragon", "Park Seo-joon", "V (BTS)"]
        }
    }
    
    local_celebs = celeb_db.get(country, {}).get(gender, ["Local Icon"])
    return headlines, local_celebs

def gemini_lookbook(country, style, headlines, celebs):
    model = genai.GenerativeModel(ACTIVE_MODEL)
    prompt = f'''
    Role: Vogue Editor for {country}.
    User Style: {json.dumps(style)}
    Trends: {headlines}
    Icons: {celebs}
    
    Output JSON only:
    {{
        "trend_summary": "Trend summary.",
        "style_translation": ["Tip 1", "Tip 2"],
        "outfit_idea": "Outfit description.",
        "shop_keywords": ["Keyword1", "Keyword2"],
        "celeb_styling": [
            {{"name": "{celebs[0]}", "wearing": "Description of them in this style"}},
            {{"name": "{celebs[1]}", "wearing": "Description"}}
        ]
    }}
    '''
    resp = generate_with_retry(model, prompt)
    return extract_json(resp.text)

# --- DATA ---
COUNTRIES = {
    "United States": {"hl": "en-US", "gl": "US", "retailers": {"Nordstrom": "[https://www.nordstrom.com/sr?keyword=](https://www.nordstrom.com/sr?keyword=){q}", "Amazon": "[https://www.amazon.com/s?k=](https://www.amazon.com/s?k=){q}"}},
    "India": {"hl": "en-IN", "gl": "IN", "retailers": {"Myntra": "[https://www.myntra.com/](https://www.myntra.com/){q}", "Ajio": "[https://www.ajio.com/search/?text=](https://www.ajio.com/search/?text=){q}"}},
    "United Kingdom": {"hl": "en-GB", "gl": "GB", "retailers": {"ASOS": "[https://www.asos.com/search/?q=](https://www.asos.com/search/?q=){q}", "Zara": "[https://www.zara.com/uk/en/search?searchTerm=](https://www.zara.com/uk/en/search?searchTerm=){q}"}},
    "France": {"hl": "fr-FR", "gl": "FR", "retailers": {"Galeries Lafayette": "[https://www.galerieslafayette.com/s/](https://www.galerieslafayette.com/s/){q}", "Zara": "[https://www.zara.com/fr/en/search?searchTerm=](https://www.zara.com/fr/en/search?searchTerm=){q}"}},
    "Italy": {"hl": "it-IT", "gl": "IT", "retailers": {"Yoox": "[https://www.yoox.com/search?dept=men&text=](https://www.yoox.com/search?dept=men&text=){q}", "Zara": "[https://www.zara.com/it/en/search?searchTerm=](https://www.zara.com/it/en/search?searchTerm=){q}"}},
    "Japan": {"hl": "ja-JP", "gl": "JP", "retailers": {"Zozotown": "[https://zozo.jp/search/?p_keyv=](https://zozo.jp/search/?p_keyv=){q}", "Rakuten": "[https://search.rakuten.co.jp/search/mall/](https://search.rakuten.co.jp/search/mall/){q}/"}},
    "South Korea": {"hl": "ko-KR", "gl": "KR", "retailers": {"MUSINSA": "[https://www.musinsa.com/search/goods?keyword=](https://www.musinsa.com/search/goods?keyword=){q}", "Coupang": "[https://www.coupang.com/np/search?component=&q=](https://www.coupang.com/np/search?component=&q=){q}"}},
}

# --- SCREEN 1: UPLOAD & PROFILE ---
def render_upload_screen():
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>LOOKBOOK AI</div>
        <div class='hero-subtitle'>Global Style Intelligence · Personalized Curation</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ PREFERENCES")
        category = st.selectbox("Collection", ["Women", "Men"])
        country = st.selectbox("Region", list(COUNTRIES.keys()))
        uploaded = st.file_uploader("Upload Outfit", type=["jpg", "png", "jpeg"])
        if st.button("↻ RESET APP", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("<div class='st-card'><h3>YOUR UPLOAD</h3>", unsafe_allow_html=True)
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)
            if st.button("✨ ANALYZE STYLE", type="primary", use_container_width=True):
                h = hashlib.md5(uploaded.getvalue()).hexdigest()
                st.session_state['style'] = gemini_style_profile(img, category, h)
                st.session_state['uploaded_file'] = uploaded
                st.session_state['category'] = category
                st.session_state['country'] = country
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if 'style' in st.session_state:
                s = st.session_state['style']
                st.markdown(f"<div class='st-card'><h3>STYLE DNA: {s.get('label', 'Undefined')}</h3>", unsafe_allow_html=True)
                
                st.markdown("**Signature Palette:**")
                pal_html = "<div style='display:flex; flex-wrap:wrap; margin-bottom:20px;'>"
                for p in s.get("expert_palette", []):
                    pal_html += f"<div style='background:{p['hex']}; width:40px; height:40px; border-radius:50%; margin-right:10px; border:1px solid #ddd;' title='{p['name']}'></div>"
                pal_html += "</div>"
                st.markdown(pal_html, unsafe_allow_html=True)
                
                st.markdown(f"**Ideal Cut:** {s.get('ideal_cut', 'Classic')}")
                
                # NAVIGATION BUTTON
                if st.button(f"🌍 TRANSLATE TO {country.upper()} (NEXT SCREEN)", type="primary", use_container_width=True):
                    with st.spinner("Analyzing trends & finding celebrity matches..."):
                        news, celebs = get_country_context(country, category)
                        st.session_state['lookbook'] = gemini_lookbook(country, s, news, celebs)
                        navigate_to('lookbook')
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👆 Click 'Analyze Style' to begin.")

# --- SCREEN 2: LOOKBOOK RESULTS ---
def render_lookbook_screen():
    country = st.session_state.get('country', 'United States')
    category = st.session_state.get('category', 'Women')
    lb = st.session_state['lookbook']
    
    # Back Button
    if st.button("← BACK TO STUDIO"):
        navigate_to('upload')
    
    st.markdown(f"<h1 style='text-align:center; font-size:3rem; margin:20px 0;'>THE {country.upper()} EDIT</h1>", unsafe_allow_html=True)
    
    # 1. Trends & Shopping
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"<div class='st-card'><h3>TREND INTELLIGENCE</h3>", unsafe_allow_html=True)
        st.info(lb.get('trend_summary'))
        for tip in lb.get('style_translation', []): st.write(f"• {tip}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='st-card'><h3>CURATED SHOPPING</h3>", unsafe_allow_html=True)
        for k in lb.get('shop_keywords', [])[:3]:
            # Gender-specific query
            query = f"{category}'s {k}"
            st.markdown(f"**{k}**")
            links = COUNTRIES[country]['retailers']
            cols = st.columns(len(links))
            for i, (name, tpl) in enumerate(links.items()):
                # PURE HTML BUTTON (Fixes localhost issue)
                url = tpl.format(q=_enc(query))
                html_btn = f'<a href="{url}" target="_blank" class="custom-btn">{name} ↗</a>'
                cols[i].markdown(html_btn, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Celebrity Photos (UPDATED LAYOUT: Name -> Image -> Desc)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; margin-bottom:30px;'>{category.upper()} ICONS IN {country.upper()}</h2>", unsafe_allow_html=True)
    
    celeb_cols = st.columns(3, gap="medium")
    for i, celeb in enumerate(lb.get('celeb_styling', [])[:3]):
        with celeb_cols[i]:
            img_url = get_celebrity_image(celeb['name'])
            
            # HTML Card: Name -> Image -> Desc
            search_q = f"{celeb['name']} {category} fashion style"
            search_url = f"[https://www.google.com/search?tbm=isch&q=](https://www.google.com/search?tbm=isch&q=){_enc(search_q)}"
            
            st.markdown(f"""
            <div class='celeb-card'>
                <div class='celeb-header'>
                    <div class='celeb-name'>{celeb['name']}</div>
                </div>
                <div class='celeb-img-box'>
                    <img src="{img_url}" alt="{celeb['name']}">
                </div>
                <div class='celeb-info'>
                    <div class='celeb-desc'>{celeb['wearing']}</div>
                    <a href="{search_url}" target="_blank" class="custom-btn custom-btn-ghost">View More Looks</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- MAIN ROUTER ---
if st.session_state.view == 'upload':
    render_upload_screen()
elif st.session_state.view == 'lookbook':
    render_lookbook_screen()