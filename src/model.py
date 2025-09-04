import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

from logger import get_logger # pylint: disable=import-error

logger = get_logger(__name__)

def cluster_function(df : pd.DataFrame)-> pd.DataFrame:
    """
    Function to cluster marks dataframe

    Args:
        df (pd.DataFrame): Dataframe to predict

    Returns:
        pd.DataFrame: Resultant dataframe
    """
    try:
        X = np.array(df['feature_vector'].to_list())
        pca_components = 100  
        pca = PCA(n_components=pca_components, random_state=42)
        X_reduced = pca.fit_transform(X)

        kmeans = MiniBatchKMeans(n_clusters=14, random_state=42, batch_size=2048, n_init="auto")
        df['cluster'] = kmeans.fit_predict(X_reduced)
        
        df['difficulty_score'] = (df['barre_chords_metric_scaled'] +
        df['major_minor_chords_metric_scaled'] + 
        df['special_chords_scaled'] + 
        df['song_length_scaled'])
        
        cluster_difficulty = df.groupby("cluster")["difficulty_score"].mean().reset_index()
        cluster_difficulty = cluster_difficulty.sort_values("difficulty_score").reset_index(drop=True)
        
        difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
        bins = pd.qcut(cluster_difficulty["difficulty_score"], q=len(difficulty_levels), labels=difficulty_levels)

        cluster_difficulty["difficulty_level"] = bins

        cluster_map = dict(zip(cluster_difficulty["cluster"], cluster_difficulty["difficulty_level"]))
        df["difficulty_level"] = df["cluster"].map(cluster_map)
        return df
    except Exception as e:
        logger.error("Error in clustering marks dataset : %s", e)
        raise

def recommend_songs(exercise_df, prod_df,exercise_id, tempo, genre, top_n=5):
    """
    Recommend top_n songs similar to the given exercise and tempo.
    Works on PCA-reduced vectors.
    """
    try:
        exercise_row = exercise_df[
            (exercise_df['exercise_id'] == exercise_id) & 
            (exercise_df['tempo'] == tempo)
        ]
        if exercise_row.empty:
            raise ValueError("No exercise found with given ID and tempo")

        exercise_vector = np.array(exercise_row['feature_vector'].iloc[0]).reshape(1, -1)
        filtered_prod_df = prod_df[prod_df['maingenre'] == genre]
        if filtered_prod_df.empty:
            raise ValueError(f"No songs found in genre '{genre}'")

        similarities = []
        for vec in filtered_prod_df['feature_vector'].values:
            sim = cosine_similarity(exercise_vector, np.array(vec, dtype=np.float32).reshape(1, -1))[0][0]
            similarities.append(sim)

        filtered_prod_df = filtered_prod_df.copy()
        filtered_prod_df['similarity'] = similarities
        top_recommendations = filtered_prod_df.sort_values(by='similarity', ascending=False).head(top_n)
        return top_recommendations[['trackname', 'artistnames', 'maingenre', 'chords', 'difficulty_level']]
    except Exception as e:
        logger.error("Error in generating recommendations : %s", e)
        raise

def recommend_songs_random(genre,songs_df, recommended_cache, n=5) -> list:
    """
    Cluster function to retrieve random songs

    Args:
        genre (_type_): String value
        n (int, optional): Number of records to retrieve Defaults to 5.

    Returns:
        list: _description_
    """
    try:
        if songs_df.empty:
            return {"error": "Dataset not loaded"}

        genre_songs = songs_df[songs_df["maingenre"] == genre]

        available_songs = genre_songs[~genre_songs["trackname"].isin(recommended_cache)]

        if available_songs.empty:
            return {"error": f"No new songs available for genre: {genre}"}

        selected = available_songs.sample(min(n, len(available_songs)), replace=False)

        recommended_cache.update(selected["trackname"].tolist())

        return selected[["trackname", "artistnames", "maingenre", "chords", 
                            "difficulty_level"]].to_dict(orient="records"),recommended_cache
    except Exception as e:
        logger.error("Error retrieving random recommendations: %s", e)
        raise

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    marks_data_file_path = os.path.join(base_dir, '..', 'data', 'processed', 'marks_data.parquet')
    exercise_data_ohe_path = os.path.join(base_dir, '..','data','processed','chord_exercises.parquet')
    marks_df = pd.read_parquet(marks_data_file_path)
    exercise_df = pd.read_parquet(exercise_data_ohe_path)
    df_prod = cluster_function(marks_df)
    df_prod_file_path = os.path.join(base_dir, '..', 'data', 'processed', 'prod_data.parquet')
    df_prod.to_parquet(df_prod_file_path)
