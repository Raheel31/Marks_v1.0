import os
import sys
import pandas as pd
from fastapi import FastAPI, Query
import uvicorn

from model import recommend_songs as model1 # pylint: disable=import-error
from logger import get_logger # pylint: disable=import-error

logger = get_logger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
df_prod_file_path = os.path.join(base_dir, '..', 'data', 'processed', 'prod_data.parquet')
exercise_df_path = os.path.join(base_dir, '..','data','processed','chord_exercises.parquet')

app = FastAPI(title="Exercise Recommendation API")

@app.get("/")
def home():
    return {"message": "Welcome to the Exercise Recommendation API"}

@app.get("/recommendations")
def recommendations(
    tempo: int = Query(..., description="Tempo value"),
    exercise_id: int = Query(..., description="Exercise ID"),
    genre: str = Query(..., description="Genre")
):
    """_summary_

    Args:
        tempo (int, optional): _description_. Defaults to Query(..., description="Tempo value").
        exercise_id (int, optional): _description_. Defaults to Query(..., description="Exercise ID").
        genre (str, optional): _description_. Defaults to Query(..., description="Genre").

    Returns:
        _type_: _description_
    """
    try:
        exercise_df = pd.read_parquet(exercise_df_path, columns=['exercise_id',
                                                             'tempo', 'feature_vector'])
        prod_data = pd.read_parquet(df_prod_file_path, columns= ['trackname', 
                                                             'artistnames', 'maingenre',
                                                             'chords', 'difficulty_level', 
                                                             'feature_vector'])
        result = model1(input_df=exercise_df, prod_df=prod_data,
                        tempo=tempo, exercise_id=exercise_id, genre=genre)
        return result
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Error fetching API: %S",e)
        return {"error": str(e)}

# ---- ENTRY POINT ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
