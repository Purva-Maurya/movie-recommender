from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_poster_cache = {}
_HEADERS = {
    "User-Agent": "MovieRecommenderApp/1.0 (personal project; contact: your-email@example.com)"
}


def fetch_poster_url(title: str):
    if title in _poster_cache:
        return _poster_cache[title]

    def _query(query_title):
        try:
            resp = requests.get(
                WIKI_SUMMARY_API + quote(query_title),
                headers=_HEADERS,
                timeout=3,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            thumbnail = data.get("thumbnail")
            return thumbnail.get("source") if thumbnail else None
        except Exception:
            return None

    url = _query(f"{title} (film)") or _query(title)
    _poster_cache[title] = url
    return url

MODEL_DIR = Path(__file__).parent / "model"

app = FastAPI(title="Movie Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "df": None,
    "indices": None,
    "tfidf_matrix": None,
}


@app.on_event("startup")
def load_model() -> None:
    df_path = MODEL_DIR / "df.pkl"
    indices_path = MODEL_DIR / "indices.pkl"
    tfidf_path = MODEL_DIR / "tfidf.pkl"

    if not (df_path.exists() and indices_path.exists() and tfidf_path.exists()):
        print(f"[WARN] Missing model files. Expected:\n  {df_path}\n  {indices_path}\n  {tfidf_path}")
        return

    df = pd.read_pickle(df_path)
    indices = pd.read_pickle(indices_path)
    vectorizer = pd.read_pickle(tfidf_path)

    df = df.reset_index(drop=True)

    text_col = "tags" if "tags" in df.columns else "overview"
    tfidf_matrix = vectorizer.transform(df[text_col].fillna(""))

    # make title lookups case-insensitive
    indices = indices.copy()
    indices.index = indices.index.str.lower()

    state["df"] = df
    state["indices"] = indices
    state["tfidf_matrix"] = tfidf_matrix
    print(f"[OK] Loaded {len(df)} movies. TF-IDF matrix shape: {tfidf_matrix.shape}")


def _require_model_loaded():
    if state["df"] is None or state["tfidf_matrix"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check df.pkl, indices.pkl, tfidf.pkl are in backend/model/.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok" if state["df"] is not None else "model_not_loaded",
        "movies_loaded": 0 if state["df"] is None else len(state["df"]),
    }


@app.get("/movies")
def search_movies(
    q: str = Query("", description="Substring to search for in titles"),
    limit: int = Query(10, ge=1, le=50),
):
    _require_model_loaded()
    df = state["df"]
    titles = df["title"]

    if q:
        mask = titles.str.contains(q, case=False, na=False, regex=False)
        titles = titles[mask]

    return titles.head(limit).tolist()


@app.get("/recommend")
def recommend(
    title: str = Query(..., description="Exact or partial movie title"),
    top_n: int = Query(10, ge=1, le=50),
):
    _require_model_loaded()
    df = state["df"]
    indices = state["indices"]
    tfidf_matrix = state["tfidf_matrix"]

    key = title.strip().lower()

    if key in indices.index:
        idx = indices[key]
        if isinstance(idx, pd.Series):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)
    else:
        partial = df[df["title"].str.lower().str.contains(key, na=False, regex=False)]
        if partial.empty:
            raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
        idx = int(partial.index[0])

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    scored = list(enumerate(sims))
    scored = sorted(scored, key=lambda pair: pair[1], reverse=True)
    scored = [pair for pair in scored if pair[0] != idx][:top_n]

    result_indices = [i for i, _ in scored]
    results = df.iloc[result_indices].copy()
    results["similarity"] = [round(float(s), 4) for _, s in scored]
    with ThreadPoolExecutor(max_workers=8) as executor:
        poster_urls = list(executor.map(fetch_poster_url, results["title"].tolist()))
    results["poster_path"] = poster_urls
    
    results = results.astype(object).where(pd.notnull(results), None)

    optional_cols = ["genres", "overview", "release_date", "poster_path", "vote_average"]
    cols = ["title"] + [c for c in optional_cols if c in results.columns] + ["similarity"]

    return {
        "query": df.iloc[idx]["title"],
        "results": results[cols].to_dict(orient="records"),
    }