from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def assign_genre_weights(genres):
    genre_list = genres.split(", ")
    weights = {genre: 1 / (i + 1) for i, genre in enumerate(genre_list)}
    return [weights.get(genre, 0) for genre in unique_genres]

books_df = books_df.dropna(subset=["Genres", "Age"])
books_df["Age"] = books_df["Age"].astype(int)
books_df[unique_genres] = books_df["Genres"].apply(assign_genre_weights).apply(pd.Series)

@app.get("/recommend")
async def recommend(genre: str, level: str):
    if level not in cefr_ages:
        return {"success": False, "message": "Invalid CEFR level."}

    age_range = cefr_ages[level]
    filtered = books_df[
        books_df["Genres"].str.contains(genre, na=False) &
        (books_df["Age"] >= age_range[0]) &
        (books_df["Age"] <= age_range[1])
    ]

    if filtered.empty:
        return {"success": True, "data": []}

    X = filtered[unique_genres + ["Age"]]
    X_normalized = X.div(np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    knn = NearestNeighbors(metric="cosine")
    knn.fit(X_normalized)

    target_vector = np.zeros(len(unique_genres) + 1)
    if genre in unique_genres:
        target_vector[unique_genres.index(genre)] = 1
    target_vector[-1] = np.mean(age_range)

    distances, indices = knn.kneighbors([target_vector], n_neighbors=len(filtered))
    result = filtered.iloc[indices[0]].copy()
    result["Distance"] = distances[0]

    return {
        "success": True,
        "data": result.sort_values("Distance")[[
            "Title", "Author", "Genres", "Age"
        ]].to_dict(orient="records")
    }
