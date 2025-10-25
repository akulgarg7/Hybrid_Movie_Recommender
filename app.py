# app.py (Corrected Version)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# --- Caching Functions to Load and Process Data Only Once ---

@st.cache_data
def load_data():
    """Loads and preprocesses the movie and ratings data."""
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    
    # --- THIS IS THE FIX ---
    # 1. Clean the '(no genres listed)' rows first.
    movies_df['genres_cleaned'] = movies_df['genres'].replace('(no genres listed)', '')
    # 2. Replace '|' correctly by disabling regex.
    movies_df['genres_processed'] = movies_df['genres_cleaned'].str.replace('|', ' ', regex=False)
    
    return movies_df, ratings_df

@st.cache_resource
def train_svd_model(ratings_df, movies_df):
    """Creates the user-item matrix and trains the SVD model."""
    # This function uses the raw 'title' column, so it doesn't need changes.
    merged_df = pd.merge(ratings_df, movies_df, on='movieId')
    user_item_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_decomposed = svd.fit_transform(user_item_matrix)
    predicted_ratings = np.dot(matrix_decomposed, svd.components_)
    
    preds_df = pd.DataFrame(predicted_ratings, 
                            index=user_item_matrix.index, 
                            columns=user_item_matrix.columns)
    return preds_df, user_item_matrix

@st.cache_resource
def calculate_similarity_matrix(_movies_df):
    """Calculates the cosine similarity matrix for content-based filtering."""
    # This function now receives the correctly processed 'genres_processed' column.
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(_movies_df['genres_processed'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# --- Main Application Logic (No changes needed below this line) ---
st.set_page_config(layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("A hybrid engine combining collaborative filtering (`TruncatedSVD`) and content-based filtering.")

# Load all data and models
movies_df, ratings_df = load_data()
preds_df, user_item_matrix = train_svd_model(ratings_df, movies_df)
cosine_sim = calculate_similarity_matrix(movies_df)
title_to_index = pd.Series(movies_df.index, index=movies_df['title'])

# --- UI Components ---
st.sidebar.header("Get Your Recommendations")
user_id = st.sidebar.selectbox("Select your User ID:", options=sorted(ratings_df['userId'].unique()))
movie_title = st.sidebar.selectbox("Select a movie you like:", options=sorted(movies_df['title'].unique()))

if st.sidebar.button("Recommend Movies"):
    # This is the final, robust recommendation logic
    CANDIDATE_POOL_SIZE = 200
    top_n = 5

    # 1. Candidate Generation (Content-Based)
    movie_idx = title_to_index[movie_title]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:CANDIDATE_POOL_SIZE + 1]
    similar_movie_indices = [i[0] for i in sim_scores]
    candidate_movies = movies_df.iloc[similar_movie_indices]

    # 2. Re-Ranking (Collaborative)
    recommendations = []
    already_rated_titles = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

    for title in candidate_movies['title']:
        if title in preds_df.columns and title not in already_rated_titles:
            predicted_rating = preds_df.loc[user_id, title]
            recommendations.append((title, predicted_rating))

    # 3. Final Output
    if recommendations:
        rec_df = pd.DataFrame(recommendations, columns=['title', 'predicted_rating'])
        rec_df = rec_df.sort_values('predicted_rating', ascending=False)
        
        st.subheader(f"Top {top_n} recommendations based on '{movie_title}':")
        st.table(rec_df.head(top_n).style.format({'predicted_rating': '{:.2f}'}))
    else:
        st.warning("Could not find any new recommendations. You might have rated all similar movies!")
else:
    st.info("Select a user and a movie, then press 'Recommend Movies' to start.")