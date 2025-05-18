# Netflix-Style Movie Recommendation System

This project implements a personalized movie recommendation system inspired by Netflix. It combines collaborative filtering (CF) using K-Nearest Neighbors (KNN) with latent features and content-based filtering (CBF) using Singular Value Decomposition (SVD). The system is deployed using Docker and Google Cloud Run for scalable and efficient serving.

## 🔍 Project Overview

- **Collaborative Filtering (CF)**: Utilizes KNN with latent features to recommend movies based on user-item interactions.
- **Content-Based Filtering (CBF)**: Employs SVD to suggest movies similar to those a user has previously liked, based on movie metadata.
- **Hybrid Approach**: Integrates CF and CBF to provide more accurate and diverse recommendations.
- **Deployment**: The model is containerized using Docker and deployed on Google Cloud Run for seamless scaling and accessibility.

## 📦 Technologies Used

- **Python**: Programming language for data processing and model development.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning algorithms and utilities.
- **Flask**: Web framework for serving the recommendation model.
- **Streamlit**: Interactive web interface for user interaction.
- **Docker**: Containerization of the application.
- **Google Cloud Run**: Serverless platform for deploying the containerized application.

## 📁 Repository Structure

```
├── deployment/             # Docker and deployment scripts
├── training/               # Model training scripts
├── main.ipynb              # Jupyter notebook for EDA and model training
├── dataset.txt             # Sample dataset
├── dataset_summary.txt     # Summary of the dataset
├── live-demo.txt           # Instructions for accessing the live demo
└── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**:

```bash
git clone https://github.com/aydiegithub/netflix-movie-recommendation-system-eda-ml-training.git
cd netflix-movie-recommendation-system-eda-ml-training
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


## 📞 Contact

**Aditya Dinesh K**  
Founder of Aydie’s Avenue  
📧 [business@aydie.in](mailto:business@aydie.in) / [aditya@aydie.in](mailto:aditya@aydie.in)  
📞 +91 9036 4694 92  
🌐 [aydie.in](https://aydie.in)
