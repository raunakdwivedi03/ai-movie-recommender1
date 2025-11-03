import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Set page config
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.sidebar.header("About")
st.sidebar.info("Content-based & Collaborative Filtering Recommendation Engine")

# TMDB API Key
TMDB_API_KEY = "YOUR_TMDB_API_KEY_HERE"  # Get from https://www.themoviedb.org/api
TMDB_BASE_URL = "https://api.themoviedb.org/3"

@st.cache_data
def load_data():
    """Load movie dataset"""
    # Using sample data - replace with your actual dataset
    movies = pd.read_csv('movies.csv')  # Should have: id, title, genres, overview, rating
    return movies

def get_movie_poster(movie_id):
    """Fetch movie poster from TMDB API"""
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            poster_path = response.json().get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

def content_based_recommendations(movie_title, df, n_recommendations=5):
    """Content-based filtering using TF-IDF and cosine similarity"""
    # Create TF-IDF matrix from movie overviews
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find movie index
    try:
        movie_idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        st.error("Movie not found!")
        return pd.DataFrame()
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genres', 'rating']]

def collaborative_filtering_recommendations(user_ratings, df, n_recommendations=5):
    """Collaborative filtering using user ratings"""
    # This is a simplified version - in production, use matrix factorization
    rating_matrix = df.set_index('title')[['rating']].copy()
    
    # Calculate similarity based on genre and rating
    user_prefs = user_ratings.groupby('genre')['rating'].mean()
    
    # Score movies based on user preferences
    df['score'] = df['genres'].apply(
        lambda x: user_prefs.get(x, 0) if isinstance(x, str) else 0
    )
    
    recommendations = df.nlargest(n_recommendations, 'score')[['title', 'genres', 'rating']]
    return recommendations

def main():
    st.header("Find Your Next Favorite Movie")
    
    # Load data
    try:
        movies = load_data()
    except FileNotFoundError:
        st.error("Please upload movies.csv file")
        return
    
    # Sidebar options
    recommendation_type = st.sidebar.radio(
        "Choose Recommendation Method:",
        ["Content-Based Filtering", "Top Rated Movies"]
    )
    
    if recommendation_type == "Content-Based Filtering":
        st.subheader("Content-Based Recommendations")
        st.write("Tell us a movie you like, and we'll find similar ones!")
        
        selected_movie = st.selectbox(
            "Select a movie you like:",
            options=sorted(movies['title'].unique())
        )
        
        n_recs = st.slider("Number of recommendations:", 3, 10, 5)
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding similar movies..."):
                recommendations = content_based_recommendations(selected_movie, movies, n_recs)
                
                if not recommendations.empty:
                    st.success(f"Movies similar to **{selected_movie}**:")
                    
                    # Display recommendations in columns
                    cols = st.columns(n_recs)
                    for idx, (col, (_, movie)) in enumerate(zip(cols, recommendations.iterrows())):
                        with col:
                            st.write(f"**{movie['title']}**")
                            st.write(f"⭐ {movie['rating']:.1f}")
                            st.write(f"Genres: {movie['genres']}")
    
    elif recommendation_type == "Top Rated Movies":
        st.subheader("Top Rated Movies")
        top_n = st.slider("Show top N movies:", 5, 20, 10)
        
        top_movies = movies.nlargest(top_n, 'rating')[['title', 'genres', 'rating']]
        
        cols = st.columns(5)
        for idx, (_, movie) in enumerate(top_movies.iterrows()):
            with cols[idx % 5]:
                st.write(f"**{movie['title']}**")
                st.write(f"⭐ {movie['rating']:.1f}")
    
    # Display dataset info
    with st.expander("Dataset Information"):
        st.write(f"Total movies: {len(movies)}")
        st.dataframe(movies.head(10))

if __name__ == "__main__":
    main()
