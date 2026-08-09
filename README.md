# 🎬 Reel Finder — Movie Recommender

A content-based movie recommendation system built with TF-IDF and cosine similarity. Search for a movie you love, get back the ones most similar to it — no external recommendation API, no paid services, no API keys anywhere in the stack.

Live demo: https://visionary-starburst-d6729c.netlify.app/ <!-- replace with your actual Netlify URL -->
API: https://movie-recommender-wlcl.onrender.com<!-- replace with your actual Render URL -->
---

## How it works

1. Each movie's metadata (title, genres, overview) is converted into a numeric vector using **TF-IDF** (Term Frequency–Inverse Document Frequency)
2. When you search a movie, the backend computes **cosine similarity** between that movie's vector and every other movie in the dataset
3. The most similar movies are returned, ranked by similarity score

No deep learning, no black box — a transparent, explainable recommendation approach.

## Features

- 🔍 Live search with autocomplete
- 🎯 Content-based recommendations ranked by similarity score
- 🖼️ Real movie posters, fetched from Wikipedia's public API (no key required)
- ⚡ Fast on-demand similarity computation — no giant precomputed matrix, scales to tens of thousands of movies
- 🎨 Custom-designed UI, no template — cinema marquee theme with a film-strip motif
- 🔓 Zero API keys, zero authentication required to run or deploy

## Tech stack

**Backend:** Python, FastAPI, scikit-learn (TF-IDF + cosine similarity), pandas
**Frontend:** Vanilla HTML/CSS/JavaScript — no framework, no build step
**Deployment:** Render (backend), Netlify (frontend)

## Project structure

```
movie-recommender/
├── backend/
│   ├── main.py              # FastAPI app — search & recommendation endpoints
│   ├── requirements.txt
│   └── model/
│       ├── df.pkl            # movie metadata dataframe
│       ├── indices.pkl       # title → row index lookup
│       └── tfidf.pkl         # fitted TF-IDF vectorizer
└── frontend/
    └── index.html            # search UI + results display
```

## Running it locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Frontend** (in a separate terminal):
```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser. Full setup walkthrough is in [`TUTORIAL.md`](TUTORIAL.md).

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Confirms the model loaded and reports dataset size |
| `GET /movies?q={query}&limit={n}` | Title search, powers autocomplete |
| `GET /recommend?title={title}&top_n={n}` | Returns the top N most similar movies |

## What I learned building this

- A "trained model" often isn't one file — mine was a vectorizer, a dataframe, and a title index, each needing different handling
- JSON can't serialize `NaN` — one missing rating value in the dataset was enough to crash the entire API
- Wikipedia's public API silently rejects requests without a proper `User-Agent` header
- Computing similarity on-demand (rather than precomputing a full similarity matrix) is the more practical approach once your dataset gets into the tens of thousands of rows — a full matrix for ~45k movies would be tens of gigabytes

## License

MIT — feel free to fork and build on this.
