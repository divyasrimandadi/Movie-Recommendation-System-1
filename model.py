import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
import pickle
import os

# Load dataset (combined movies and ratings dataset)
data_file = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\indian movies.csv"
df = pd.read_csv(data_file)

# Ensure correct column names
required_columns = {"movieId", "title", "year", "language", "rating", "genres", "Votes", "Timing"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Convert rating to numeric (handle possible string issues)
df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
df["rating"] = df["rating"].fillna(df["rating"].mean())

# Fill missing values in Votes and Timing
df["Votes"] = df["Votes"].fillna(0)
df["Timing"] = df["Timing"].fillna("Unknown")

# Handle duplicate movieId-title pairs by averaging ratings
df = df.groupby(["movieId", "title"]).agg({"rating": "mean"}).reset_index()

# Create a movie-item interaction matrix
movie_item_matrix = df.pivot(index="movieId", columns="title", values="rating").fillna(0)

# Convert to sparse matrix
sparse_matrix = sparse.csr_matrix(movie_item_matrix.values)

# Train ALS Model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(sparse_matrix)

# Ensure models directory exists
model_dir = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "als_model.pkl")

# Save model for later use
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model training complete. Saved as {model_path}")
