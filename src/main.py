import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
import uvicorn
import pyarrow.dataset as ds

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import recommend_songs as model1  # pylint: disable=import-error
from logger import get_logger  # pylint: disable=import-error

logger = get_logger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data", "processed")

prod_file = os.path.join(data_dir, "prod_data.parquet")
exercise_file = os.path.join(data_dir, "chord_exercises.parquet")
recommended_history = set()

app = FastAPI(title="Exercise Recommendation API")

def read_prod_in_batches(file_path, genre, columns,batch_size=500):
    """Generator to read Parquet prod data in batches filtered by genre."""
    dataset = ds.dataset(file_path, format="parquet")
    filter_expr = (ds.field("maingenre") == genre)
    scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size, columns=columns)

    for record_batch in scanner.to_batches():
        df = record_batch.to_pandas()
        df['feature_vector'] = df['feature_vector'].apply(lambda x: np.array(x, dtype=np.float16))
        yield df

@app.get("/")
def home():
    return {"message": "Welcome to the Exercise Recommendation API"}

@app.get("/random_exercises")
def random_exercises(genre: str = Query(..., description="Genre of exercises"), n: int = 5):
    """Return n random exercises in batches to reduce memory usage."""
    try:
        results = []
        recommended_temp = set()

        for batch_df in read_prod_in_batches(prod_file, genre, batch_size=500,
                                             columns=["trackname", "artistnames", "maingenre", 
                                                      "chords", "difficulty_level","feature_vector"]):
            available_songs = batch_df[~batch_df["trackname"].isin(recommended_history)]
            if not available_songs.empty:
                selected = available_songs.sample(min(n - len(results), len(available_songs)))
                results.append(selected)
                recommended_temp.update(selected["trackname"].tolist())
            if len(results) >= n:
                break

        if not results:
            return {"error": f"No new songs available for genre: {genre}"}

        final_df = pd.concat(results)
        recommended_history.update(recommended_temp)

        return final_df[["trackname", "artistnames", "maingenre", "chords", "difficulty_level"]].to_dict(orient="records")
    except Exception as e:
        logger.error("Error fetching API: %s", e)
        return {"error": str(e)}


@app.get("/recommendations")
def recommendations(
    tempo: int = Query(..., description="Tempo value"),
    exercise_id: int = Query(..., description="Exercise ID"),
    genre: str = Query(..., description="Genre"),
):
    """Return top N recommended songs for a given exercise and tempo using batch processing."""
    try:
        exercise_df = pd.read_parquet(
            exercise_file,
            engine="pyarrow",
            filters=[("exercise_id", "=", exercise_id)],
        )
        prod_batches = []
        for batch_df in read_prod_in_batches(prod_file, genre, batch_size=500, 
                                             columns=["trackname", "artistnames",
                                                      "maingenre", "chords",
                                                      "difficulty_level","feature_vector"]):
            prod_batches.append(batch_df)
        prod_df = pd.concat(prod_batches, ignore_index=True)

        result = model1(
            exercise_df=exercise_df,
            prod_df=prod_df,
            tempo=tempo,
            exercise_id=exercise_id,
            genre=genre,
        )
        return result

    except Exception as e:
        logger.error("Error fetching API: %s", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
