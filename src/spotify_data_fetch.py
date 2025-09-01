import os
import time
import random
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from logger import get_logger # pylint: disable=import-error

logger = get_logger(__name__)

def read_spotify_track_data(track_ids: list,
                            output_file: str,
                            save_every: int = 5000,
                            max_records: int = 25000) -> pd.DataFrame:
    """
    Fetches Spotify track data in batches with a tiny delay between batches to avoid rate limits.

    Args:
        track_ids (list): Spotify track ids
        save_every (int): Save checkpoint after this many records
        output_file (str): File path for partial results
        max_records (int): Max number of tracks to fetch in one run

    Returns:
        pd.DataFrame: Track data (song, artist, id, etc.)
    """
    try:
        client_id = "754888cc4fa4486daea9cb7917e176fc"
        client_secret = "3f383e012f7442c18851a668b63849dc"

        sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
        )

        # Load partial results if they exist
        if os.path.exists(output_file):
            existing = pd.read_parquet(output_file)
            processed_ids = set(existing["spotify_song_id"])
        else:
            existing = pd.DataFrame()
            processed_ids = set()

        # Filter only unprocessed IDs
        remaining_ids = [tid for tid in track_ids if tid not in processed_ids]

        # Limit to max_records this run
        remaining_ids = remaining_ids[:max_records]

        results = []
        total = len(remaining_ids)

        for i in range(0, total, 50):
            batch = remaining_ids[i:i+50]
            response = sp.tracks(batch)

            for track in response["tracks"]:
                if track is None:
                    continue
                track_id = track["id"]
                track_name = track["name"]
                artist_names = ", ".join([artist["name"] for artist in track["artists"]])
                results.append({
                    "spotify_song_id": track_id,
                    "track_name": track_name,
                    "artist_names": artist_names
                })

            # Tiny random delay to avoid hitting rate limit
            time.sleep(random.uniform(0.2, 0.5))

            if (i // 50) % (save_every // 50) == 0 and results:
                df_partial = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
                df_partial.to_parquet(output_file, index=False)

        # Final save
        final_df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
        final_df.to_parquet(output_file, index=False)
        print(f"Run complete. Total saved: {len(final_df)} tracks")

        return final_df
    except Exception as e:
        logger.error("Failed fetching spotify data : %s", e)
        raise

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "..", "data", "raw", "songs_data.parquet")

    chord_data = pd.read_parquet(file_path)
    track_id_list = [str(id) for id in chord_data["spotify_song_id"] if str(id).lower() != 'none']
    output_path = os.path.join(BASE_DIR, "..", "data", "raw", "spotify_tracks.parquet")
    spotify_track_data = read_spotify_track_data(track_id_list, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spotify_track_data.to_parquet(output_path, index=False)
