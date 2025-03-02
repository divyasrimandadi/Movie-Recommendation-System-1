from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse

app = Flask(__name__)

# Load dataset
data_file = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\indian movies.csv"
df = pd.read_csv(data_file)
print("Dataset Columns:", df.columns)

# Load trained ALS model
model_path = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\models\\als_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Create mapping of movieId to title
movie_mapping = dict(zip(df["movieId"], df["title"]))

@app.route("/recommend", methods=["GET"])
def recommend_movies():
    try:
        movie_id = int(request.args.get("movieId"))
        num_recommendations = int(request.args.get("num", 5))

        if movie_id not in movie_mapping:
            return jsonify({"error": "Invalid movieId"}), 400

        # Convert movie ID to interaction matrix index
        movie_index = df.index[df["movieId"] == movie_id].tolist()[0]

        # Get similar movie recommendations
        similar_movies = model.similar_items(movie_index, N=num_recommendations + 1)

        recommendations = [
            {"movieId": int(df.iloc[i]["movieId"]), "title": df.iloc[i]["title"]}
            for i, _ in similar_movies[1:]
        ]

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
