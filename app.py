from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data_file = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\indian movies.csv"
df = pd.read_csv(data_file)

# Ensure correct column names
required_columns = {"movieId", "title", "year", "language", "rating", "genres", "Votes", "Timing"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Convert rating to numeric and handle missing values
df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
df["rating"] = df["rating"].fillna(df["rating"][df["rating"].notna()].mean())  # Replace NaN with the mean of valid ratings

df["Votes"] = df["Votes"].fillna(0)
df["Timing"] = df["Timing"].fillna("Unknown")

# Load trained model
model_path = "C:\\Users\\Divya\\OneDrive\\Documents\\Movie Recommendation System-1\\models\\als_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/recommend', methods=['POST'])
def recommend():
    genre = request.form.get('genre', '').strip()
    language = request.form.get('language', '').strip()
    year = request.form.get('year', '').strip()
    rating = request.form.get('rating', '').strip()
    
    filtered_df = df.copy()  # Start with the full dataset

    # Apply filters only if the user provided a value
    if genre:
        filtered_df = filtered_df[df["genres"].astype(str).str.contains(genre, case=False, na=False)]
    if language:
        filtered_df = filtered_df[df["language"].astype(str).str.contains(language, case=False, na=False)]
    if year.isdigit():
        filtered_df = filtered_df[df["year"] == int(year)]
    if rating.replace('.', '', 1).isdigit():
        filtered_df = filtered_df[df["rating"] >= float(rating)]

    # Get top 10 recommendations sorted by rating
    recommended_movies = filtered_df[["title", "year", "rating"]].sort_values(by="rating", ascending=False).head(10)

    return render_template('recommend.html', movies=recommended_movies.to_dict(orient='records'))



if __name__ == '__main__':
    app.run(debug=True)
