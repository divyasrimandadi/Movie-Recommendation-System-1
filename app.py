from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get user input from index.html
    user_input = request.form.get("movie_name")

    # Sample recommendations 
    recommended_movies = [
        ("Inception", "https://image.tmdb.org/t/p/w500/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg"),
        ("Interstellar", "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"),
        ("The Dark Knight", "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg")
    ]

    return render_template("recommend.html", movies=recommended_movies, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
