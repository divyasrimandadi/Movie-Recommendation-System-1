from flask import Flask, render_template, request
from model import load_data, build_model, recommend_movies
import os


app = Flask(__name__)


df = load_data()
model, movie_to_idx, df = build_model(df)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    genre = request.form.get("genre")
    year = request.form.get("year")
    language = request.form.get("language")
    rating = request.form.get("rating")
    
    # Convert inputs
    year = int(year) if year else None
    rating = float(rating) if rating else 0
    
    movies = recommend_movies(model, movie_to_idx, df, genre=genre, year=year, language=language, min_rating=rating)
    
    return render_template("recommend.html", movies=[
    {"title": m[0], "rating": m[1], "imdb_id": df[df["title"] == m[0]]["movieId"].values[0]}
    for m in movies
])



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
