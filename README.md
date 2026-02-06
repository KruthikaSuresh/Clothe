# LOOKBOOK AI  
### Global Style Intelligence Â· Personalized Fashion Curation

LOOKBOOK AI is a Streamlit-based fashion intelligence application that analyzes an uploaded outfit photo to generate a **personalized color palette, style insight, and region-specific lookbook**.

The system follows an **offline-first architecture**, using local computer vision and color analysis for core functionality, and optionally enhancing results with **Google Gemini** for higher-level style reasoning and curated recommendations.

---

## Key Features

### Style & Color Analysis (Offline-First)
- Dominant color palette extraction from the outfit
- Optional face detection for complexion-aware analysis
- Undertone (warm / cool / neutral) and depth estimation
- Morning vs Evening palette recommendations
- Graceful fallback to outfit-only analysis if no face is detected

### Gemini-Powered Enhancements (Optional)
- Structured style insights from the uploaded photo
- Outfit summary, style tags, and color pairing advice
- Regional lookbook generation using fashion trends
- Curated shopping keywords aligned with palette & context
- Automatic Gemini model resolution (avoids unsupported models)
- API key rotation to handle quota and rate limits

### Regional Fashion Intelligence
- Country-specific fashion trends via Google News RSS
- Cultural style grounding using representative fashion icons
- Region-aware retailer links (US, India, UK, Japan)
- No celebrity images stored or displayed (links only)

### Curated Shopping Experience
- Morning / Evening shopping modes
- Palette-driven item search
- Google Shopping & Pinterest discovery links
- Retailer-specific site search (e.g. Zara, Myntra, ASOS)

---

## Tech Stack

- Python 3.9+
- Streamlit
- Google Gemini (google-generativeai)
- Pillow
- NumPy
- OpenCV (optional)
- feedparser
- python-dotenv

---

## Project Structure

```
.
â”œâ”€â”€ app.py                # Main Streamlit application
â”œâ”€â”€ .env                  # Environment variables (not committed)
â”œâ”€â”€ requirements.txt      # Dependencies
â””â”€â”€ README.md
```

---

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

**Windows**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit google-generativeai pillow feedparser requests python-dotenv numpy opencv-python
```

> OpenCV is optional. If unavailable, the app automatically falls back to outfit-only analysis.

---

### 3. Configure Environment Variables

Create a `.env` file in the project root.

**Single API key**
```env
GEMINI_API_KEY=your_api_key_here
```

**Multiple API keys (recommended)**
```env
GEMINI_API_KEYS=key1,key2,key3
```

---

## Running the Application

```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## How It Works (Technical Overview)

1. User uploads an outfit image
2. Local pipeline extracts palette and complexion signals
3. Gemini (if enabled) generates structured style insights
4. Regional lookbook is created using trends and context
5. Palette-aware shopping links are generated dynamically

---

## Design Principles

- Offline-first and resilient to API failures
- Graceful degradation when Gemini or OpenCV is unavailable
- Copyright-safe (no celebrity images stored)
- Transparent AI usage (model shown in UI)
- Performance-aware caching

---

## Security Notes

- API keys loaded via environment variables or Streamlit secrets
- `.env` must never be committed to version control
- No user images are stored on disk

---

## License

Add a license before public release  
(MIT or Apache-2.0 recommended)
