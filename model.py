import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix

# Load dataset
def load_data(file_path="indian movies.csv"):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["rating", "year"])  # Remove missing ratings
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])  # Drop remaining NaNs
    
    return df

def build_model(df):
    # Assign unique user IDs for training (assume movies are user interactions)
    df["user_id"] = np.arange(len(df))
    
    # Create a sparse matrix
    movie_to_idx = {movie: i for i, movie in enumerate(df["title"].unique())}
    df["movie_idx"] = df["title"].map(movie_to_idx)
    user_movie_matrix = csr_matrix((df["rating"], (df["user_id"], df["movie_idx"])))
    
    # Train ALS model
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    model.fit(user_movie_matrix.T)
    
    return model, movie_to_idx, df

def recommend_movies(model, movie_to_idx, df, genre=None, year=None, language=None, min_rating=0):
    movie_scores = {}
    for movie, idx in movie_to_idx.items():
        movie_data = df[df["title"] == movie].iloc[0]
        if genre and genre not in movie_data["genres"]:
            continue
        if year and movie_data["year"] != str(year):
            continue
        if language and movie_data["language"] != language.lower():
            continue
        if movie_data["rating"] < min_rating:
            continue
        
        movie_scores[movie] = movie_data["rating"]
    
    # Sort by rating
    return sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]

if __name__ == "__main__":
    df = load_data()
    model, movie_to_idx, df = build_model(df)
    recommendations = recommend_movies(model, movie_to_idx, df, genre="Action", year=2020, min_rating=7)
    print(recommendations)
