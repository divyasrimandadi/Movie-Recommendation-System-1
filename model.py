import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
import pickle
import os

# Load dataset
data_file = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\indian movies.csv"
df = pd.read_csv(data_file)

# Ensure correct column names
required_columns = {"movieId", "title", "year", "language", "rating", "genres", "Votes", "Timing"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Convert rating to numeric and handle missing values
df["rating"] = pd.to_numeric(df["rating"], errors='coerce').fillna(0)

df["Votes"] = df["Votes"].fillna(0)
df["Timing"] = df["Timing"].fillna("Unknown")

# Handle duplicate movieId-title pairs by averaging ratings
df = df.groupby(["movieId", "title"]).agg({"rating": "mean"}).reset_index()

# Create movie-item interaction matrix
movie_item_matrix = df.pivot(index="movieId", columns="title", values="rating").fillna(0)

# Convert to sparse matrix
sparse_matrix = sparse.csr_matrix(movie_item_matrix.values)

# Train ALS Model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(sparse_matrix)

# Save model
model_dir = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "als_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model training complete. Saved as {model_path}")
