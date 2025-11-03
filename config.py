import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "your_api_key_here")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Application Configuration
APP_TITLE = "Movie Recommendation System"
APP_DESCRIPTION = "Content-based & Collaborative Filtering Recommendation Engine"

# Recommendation Parameters
MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS = 5
MIN_RECOMMENDATIONS = 3

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = 'english'

# Data Configuration
DATA_PATH = "data/movies.csv"
CACHE_ENABLED = True

# API Configuration
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

# Streamlit Configuration
LAYOUT = "wide"
THEME = "light"
