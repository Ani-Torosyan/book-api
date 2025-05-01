from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pydantic import BaseModel

app = FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataset and prepare it globally
books_df = pd.read_csv("books_with_genres.csv", encoding="utf-8-sig")
cefr_ages = {
    "A1": (0, 7), "A2": (8, 11), "B1": (12, 15),
    "B2": (16, 17), "C1": (18, 25)
}

unique_genres = sorted(set(g for genres in books_df["Genres"].dropna() for g in genres.split(", ")))


class RecommendationRequest(BaseModel):
    genre: str  # Single genre selected by the user
    level: str  # Single CEFR level selected by the user

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    genre = request.genre
    level = request.level

    # Validate the CEFR level
    if level not in cefr_ages:
        return {"success": False, "message": f"Invalid CEFR level: {level}"}

    # Get the age range for the provided level
    min_age, max_age = cefr_ages[level]

    # Filter books based on the genre and the age range
    filtered = books_df[
        books_df["Genres"].apply(lambda x: genre in x) &
        (books_df["Age"] >= min_age) &
        (books_df["Age"] <= max_age)
    ]

    if filtered.empty:
        return {"success": True, "data": []}

    # Prepare the features for KNN (including genre weights and age)
    X = filtered[unique_genres + ["Age"]]
    X_normalized = X.div(np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    knn = NearestNeighbors(metric="cosine")
    knn.fit(X_normalized)

    # Create a target vector based on the selected genre
    target_vector = np.zeros(len(unique_genres) + 1)
    if genre in unique_genres:
        target_vector[unique_genres.index(genre)] = 1
    target_vector[-1] = (min_age + max_age) / 2  # Midpoint of the age range

    distances, indices = knn.kneighbors([target_vector], n_neighbors=len(filtered))
    result = filtered.iloc[indices[0]].copy()
    result["Distance"] = distances[0]

    return {
        "success": True,
        "data": result.sort_values("Distance")[["Title", "Author", "Genres", "Age", "Image URL", "Book URL"]].to_dict(orient="records")
    }