import os
import requests

# Hardcode or load from .env

TMDB_API_KEY = "adde2ee0b726cf2449d551c770c694f0"

def fetch_tmdb_info(tmdb_id):
    """Fetch poster URL and overview for a given TMDb movie ID."""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    poster = f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}"
    return {
        "title": data.get("title"),
        "overview": data.get("overview"),
        "poster_url": poster,
        "genres": [g["name"] for g in data.get("genres", [])]
    }