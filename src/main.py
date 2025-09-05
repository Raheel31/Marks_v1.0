import os
import sys
import pandas as pd
from fastapi import FastAPI, Query
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import recommend_songs as model1  # pylint: disable=import-error
from model import recommend_songs_random as model2  # pylint: disable=import-error
from logger import get_logger  # pylint: disable=import-error

logger = get_logger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data", "processed")

prod_file = os.path.join(data_dir, "prod_data.parquet")
exercise_file = os.path.join(data_dir, "chord_exercises.parquet")
recommended_history = set()

app = FastAPI(title="Exercise Recommendation API")

@app.get("/")
def home():
    return {"message": "Welcome to the Exercise Recommendation API"}

@app.get("/random_exercises")
def random_exercises(genre: str = Query(..., description="Genre of exercises")):
    """Return n random exercises in batches to reduce memory usage."""
    try:
        recommended_temp = set()
        prod_df = pd.read_parquet(
            prod_file,
            filters=[("maingenre", "=", genre)])
        result = model2(genre=genre, songs_df=prod_df, recommended_cache=recommended_temp)

        recommended_history.update(recommended_temp)
        return result
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
            filters=[("exercise_id", "=", exercise_id)],
        )
        prod_df = pd.read_parquet(
            prod_file,
            filters=[("maingenre", "=", genre)])

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
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
