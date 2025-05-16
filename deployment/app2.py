import os
import pickle
import joblib
import polars as pl
import streamlit as st

from sklearn.neighbors import NearestNeighbors
from surprise import SVD

from utils.tmdb import fetch_tmdb_info

st.set_page_config(page_title="🍿 Netflix-Style Recommender", layout="wide")

@st.cache_resource
def load_models():
    # Load latent factors (pandas DataFrame indexed by movieId)
    with open("models/latent_df.pkl", "rb") as f:
        latent_df = pickle.load(f)
    # Load NearestNeighbors for latent factors
    nn_latent = joblib.load("models/nn_latent.joblib")
    # Load SVD (for personalized rating if desired)
    svd = joblib.load("models/svd.joblib")
    return latent_df, nn_latent, svd

@st.cache_data
def load_movies():
    # MovieLens movies.parquet, must include tmdbId column
    return pl.read_parquet("data/movies.parquet")[['movieId', 'title', 'tmdbId', 'genres']]

latent_df, nn_latent, svd = load_models()
movies = load_movies()

# Build mapping from movieId to latent_df index
id_to_idx = {mid: idx for idx, mid in enumerate(latent_df.index)}
idx_to_mid = {v: k for k, v in id_to_idx.items()}

st.title("🍿 Netflix-Style Movie Recommender")
st.write("Search any movie to see similar titles and predicted ratings.")

# --- Sidebar for personalization (optional) ---
with st.sidebar:
    st.header("Customize")
    n_similar = st.slider("How many similar?", 5, 20, 10)
    per_genre = st.slider("Per-genre picks", 1, 5, 5)

# --- Main search bar ---
movie_input = st.text_input("Enter movie title:", "Avatar")
if st.button("Recommend"):
    # 1) Lookup movie
    match = movies.filter(
        pl.col("title").str.to_lowercase().str.contains(movie_input.lower())
    )
    if match.height == 0:
        st.error(f"No movie found for '{movie_input}'")
        st.stop()

    # Take first match
    row = match.row(0)
    mid = int(row[0])  # movieId
    tmdb_id = int(row[2])  # tmdbId

    # 2) Predict rating via SVD
    try:
        pred = svd.predict(1000, mid).est
    except:
        pred = latent_df.loc[mid].mean()  # fallback: global mean of latent factors
    st.subheader(f"Predicted Rating: ⭐ {pred:.2f}")

    # 3) Fetch TMDb metadata
    info = fetch_tmdb_info(tmdb_id)
    st.markdown(f"**{info['title']}**")
    st.image(info["poster_url"], width=200)
    st.write(info["overview"])

    # 4) Find nearest neighbors in latent space
    vec = latent_df.loc[mid].values.reshape(1, -1)
    dists, inds = nn_latent.kneighbors(vec, n_neighbors=n_similar+1)
    rec_ids = [idx_to_mid[i] for i in inds[0] if idx_to_mid[i] != mid][:n_similar]

    # 5) Display similar movies as a grid
    st.markdown("### Similar Movies")
    cols = st.columns(5)
    for i, rec_mid in enumerate(rec_ids):
        with cols[i % 5]:
            rec_row = movies.filter(pl.col("movieId") == rec_mid).row(0)
            rec_info = fetch_tmdb_info(int(rec_row[2]))
            st.image(rec_info["poster_url"], caption=rec_info["title"], use_container_width=True)

    # 6) Genre-filtered picks
    st.markdown("### Top Picks by Genre")
    genres = row[3].split("|")
    for g in genres[:3]:  # show up to 3 genres
        st.markdown(f"**{g}**")
        picks = (
            movies
            .filter(pl.col("genres").str.contains(g))
            .sample(per_genre, with_replacement=False, seed=42)
        )
        cols = st.columns(per_genre)
        for j, pick in enumerate(picks.rows()):
            with cols[j]:
                pi = fetch_tmdb_info(int(pick[2]))
                st.image(pi["poster_url"], caption=pi["title"], use_container_width=True)