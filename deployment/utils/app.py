import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from utils.tmdb import fetch_tmdb_info

st.set_page_config(page_title="🍿 Netflix Movie Recommendation", layout="wide")

# 1) Load data
movies       = pd.read_csv('data/movies.csv')
links        = pd.read_csv('data/links.csv')
ratings      = pd.read_csv('data/ratings.csv')
movie_feats  = pd.read_csv('data/train.csv')  # contains movie_rating_* columns

# 2) Merge and prepare
movies = movies.merge(links[['movieId','tmdbId']], on='movieId', how='left')
nfeat_cols = ['movie_rating_count','movie_rating_mean','movie_rating_var','movie_rating_std']
X_movies    = movie_feats[nfeat_cols].values

# 3) Fit NearestNeighbors
nn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
nn.fit(X_movies)

# 4) Load the rating prediction model
rating_model = joblib.load('models/xgb_rating_model.joblib')

st.title("🍿 Netflix Movie Recommender")
st.write("Choose a movie and get recommendations instantly.")

# Sidebar
with st.sidebar:
    user_id   = st.number_input("Your User ID", min_value=1, value=1)
    n_similar = st.slider("How many similar movies?", 5, 10, 5)
    per_genre = st.slider("Per-genre picks", 1, 3, 1)

# Main selector
movie_input = st.selectbox("Select a movie:", movies['title'].tolist())

if st.button("Recommend"):
    row     = movies[movies.title == movie_input].iloc[0]
    mid     = int(row.movieId)
    tmdb_id = int(row.tmdbId)

    # Predict rating
    mf = movie_feats[movie_feats.movieId == mid]
    Xq = mf[nfeat_cols]
    pred = rating_model.predict(Xq)[0]
    st.subheader(f"Predicted Rating: ⭐ {pred:.2f}")

    # TMDb details
    info = fetch_tmdb_info(tmdb_id)
    st.image(info['poster_url'], width=200)
    st.write(info['overview'])

    # Similar movies
    idx       = movie_feats.index[movie_feats.movieId == mid][0]
    distances, indices = nn.kneighbors(X_movies[idx].reshape(1,-1), n_neighbors=n_similar+1)
    sim_ids   = movie_feats.iloc[indices[0][1:]].movieId.values
    sim_titles= movies[movies.movieId.isin(sim_ids)].title.tolist()

    st.markdown("### Similar Movies")
    for title in sim_titles:
        st.write(f"- {title}")

    # Genre-based picks
    st.markdown("### Genre-based Picks")
    genres = row.genres.split('|')
    for g in genres[:per_genre]:
        st.write(f"**{g}**")
        candidates = movie_feats[movie_feats[g] == 1]
        topn = candidates.sort_values('movie_rating_mean', ascending=False).movieId.head(3)
        picks = movies[movies.movieId.isin(topn)].title.tolist()
        for p in picks:
            st.write(f"- {p}")