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
    allow_origins=["https://armling.vercel.app/recommendations/books"],  # use specific domain in production
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
    genres: list[str]  # List of genres selected by the user
    levels: list[str]  # List of CEFR levels selected by the user


def assign_genre_weights(genres):
    genre_list = genres.split(", ")
    weights = {genre: 1 / (i + 1) for i, genre in enumerate(genre_list)}
    return [weights.get(genre, 0) for genre in unique_genres]

books_df = books_df.dropna(subset=["Genres", "Age"])
books_df["Age"] = books_df["Age"].astype(int)
books_df[unique_genres] = books_df["Genres"].apply(assign_genre_weights).apply(pd.Series)


@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    genres = request.genres
    levels = request.levels

    # Validate CEFR levels
    for level in levels:
        if level not in cefr_ages:
            return {"success": False, "message": f"Invalid CEFR level: {level}"}

    if not levels:
        return {"success": False, "message": "No CEFR levels provided."}

    # Prepare the combined age range based on levels
    age_range = [cefr_ages[level] for level in levels]

    # If age_range is empty, return an error
    if not age_range:
        return {"success": False, "message": "Age range could not be determined from the provided levels."}

    min_age = min([age[0] for age in age_range])
    max_age = max([age[1] for age in age_range])

    # Filter books based on genres and the combined age range
    filtered = books_df[ 
        books_df["Genres"].apply(lambda x: any(g in x for g in genres)) & 
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

    # Create a target vector based on the selected genres
    target_vector = np.zeros(len(unique_genres) + 1)
    for genre in genres:
        if genre in unique_genres:
            target_vector[unique_genres.index(genre)] = 1
    target_vector[-1] = (min_age + max_age) / 2  # Midpoint of the age range

    distances, indices = knn.kneighbors([target_vector], n_neighbors=len(filtered))
    result = filtered.iloc[indices[0]].copy()
    result["Distance"] = distances[0]

    return {
        "success": True,
        "data": result.sort_values("Distance")[["Title", "Author", "Genres", "Age"]].to_dict(orient="records")
    }
