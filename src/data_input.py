import pandas as pd
from logger import get_logger # pylint: disable=import-error

logger = get_logger(__name__)

def read_chord_data(url : str) -> pd.DataFrame:
    """
    Function to read songs chord ata 

    Args:
        url (str): Web api link to chord dataset

    Returns:
        pd.DataFrame: Pandas read dataframe
    """
    try:
        logger.info('Reading Chord Data')
        dataset = pd.read_csv(url, dtype=str)
        return dataset
    except Exception as e:
        logger.error("Error reading chord data : %s", e)
        raise

if __name__ == '__main__':
    chord_data_url = "hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv"
    df = read_chord_data(url= chord_data_url)
    df.to_parquet("data/raw/songs_data.parquet", engine="pyarrow", index=False)
