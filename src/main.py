import os
import sys
import pandas as pd
from fastapi import FastAPI, Query
import uvicorn
from model import recommend_songs as model1  # pylint: disable=import-error
from model import recommend_songs_random as model2 # pylint: disable=import-error
from logger import get_logger  # pylint: disable=import-error


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    try:
        logger.info("Reading prod data")
        prod_df = pd.read_parquet(
            prod_file,
            engine="pyarrow",
            filters=[("maingenre", "=", genre)]
        )
        result, recommended_history_temp = model2(genre, songs_df=prod_df, recommended_cache=recommended_history)
        recommended_history.update(recommended_history_temp)
        return {"genre": genre, "recommendations": result}
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Error fetching API: %S",e)
        return {"error": str(e)}

@app.get("/recommendations")
def recommendations(
    tempo: int = Query(..., description="Tempo value"),
    exercise_id: int = Query(..., description="Exercise ID"),
    genre: str = Query(..., description="Genre")
):
    try:
        logger.info("Reading prod data")
        prod_df = pd.read_parquet(
            prod_file,
            engine="pyarrow",
            filters=[("maingenre", "=", genre)],
            columns=['trackname', 'artistnames', 'maingenre', 'chords', 'difficulty_level','feature_vector']
        )
        exercise_df = pd.read_parquet(
            exercise_file,
            engine="pyarrow",
            filters=[("exercise_id", "=", exercise_id)]
        )
        result = model1(
            exercise_df=exercise_df,
            prod_df=prod_df,
            tempo=tempo,
            exercise_id=exercise_id,
            genre=genre
        )
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error fetching API: %s", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
