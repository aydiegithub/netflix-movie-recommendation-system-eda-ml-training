# 🎬 Netflix-Style Movie Recommendation System

This project is a personalized movie recommendation system inspired by Netflix. It leverages collaborative filtering techniques, integrates with The Movie Database (TMDB) API for enriched movie details, and offers both Streamlit and Flask web interfaces for user interaction.

---

## 📌 Features

- ^[**Data Preprocessing**: Cleaned and merged datasets including movie details, user ratings, and tags.]({"attribution":{"attributableIndex":"460-1"}})
- ^[**Feature Engineering**: Applied one-hot encoding to movie tags for genre classification.]({"attribution":{"attributableIndex":"460-2"}})
- ^[**Model Training**: Utilized the Surprise library's Singular Value Decomposition (SVD) for collaborative filtering.]({"attribution":{"attributableIndex":"460-3"}})
- ^[**Model Export**: Saved trained models and dataframes for efficient loading and inference.]({"attribution":{"attributableIndex":"460-4"}})
- ^[**TMDB Integration**: Fetched movie posters, descriptions, and crew information using TMDB API.]({"attribution":{"attributableIndex":"460-5"}})
- ^[**Web Interface**: Developed user-friendly interfaces using Streamlit and Flask.]({"attribution":{"attributableIndex":"460-6"}})
- ^[**Deployment**: Deployed the application on Google Cloud Run using Docker.]({"attribution":{"attributableIndex":"460-7"}})

---

## 🗂️ Project Structure

```
netflix-recommender/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── movies.parquet
├── models/
│   ├── SVD.joblib
│   ├── NNLatent.joblib
│   └── NNLatentDataFrame.pkl
├── app/
│   ├── main.py
│   ├── tmdb.py
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```



---

## ⚙️ Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/netflix-recommender.git
   cd netflix-recommender
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set TMDB API Key**:

   Obtain an API key from [TMDB](https://www.themoviedb.org/documentation/api) and set it as an environment variable:

   ```bash
   export TMDB_API_KEY='your_api_key'  # On Windows: set TMDB_API_KEY=your_api_key
   ```

---

## 🚀 Running the Application

### Option 1: Streamlit Interface

^[Provides an interactive web interface for movie recommendations.]({"attribution":{"attributableIndex":"1513-8"}})

```bash
streamlit run app/streamlit_app.py
```

- ^[Access the app at `http://localhost:8501/`.]({"attribution":{"attributableIndex":"2303-0"}})

### Option 2: Flask Interface

^[Offers a web application using Flask framework.]({"attribution":{"attributableIndex":"2401-1"}})

```bash
python app/main.py
```

- ^[Access the app at `http://localhost:5000/`.]({"attribution":{"attributableIndex":"2484-0"}})

---

## 📦 Deployment on Google Cloud Run

1. **Build Docker Image**:

   ```bash
   docker build -t netflix-recommender .
   ```

2. **Deploy to Google Cloud Run**:

   Follow [Google Cloud Run deployment guide](https://cloud.google.com/run/docs/deploying) to deploy the Docker image.

3. **Set Environment Variables**:

   Ensure `TMDB_API_KEY` is set in the Cloud Run service settings.

4. **Access the Application**:

   Once deployed, access the application via the URL provided by Google Cloud Run.

---

## 📝 Acknowledgments

- ^[[TMDB API](https://www.themoviedb.org/documentation/api) for movie data and images.]({"attribution":{"attributableIndex":"2566-9"}})
- ^[[Surprise Library](http://surpriselib.com/) for building the recommendation engine.]({"attribution":{"attributableIndex":"2566-10"}})
- ^[[Streamlit](https://streamlit.io/) and [Flask](https://flask.palletsprojects.com/) for web interfaces.]({"attribution":{"attributableIndex":"2566-11"}})
