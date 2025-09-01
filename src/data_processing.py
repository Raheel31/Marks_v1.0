import os
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from logger import get_logger # pylint: disable=import-error


logger = get_logger(__name__)

def clean_string(vars : list) -> list: # pylint: disable=redefined-builtin
    """
    Funciton to manipulate and "clean" string values

    Args:
        vars (list): list of string variables to manipulate

    Returns:
        list: cleaned list of strings
    """
    try:
        clean_vars = []
        if isinstance(vars, str):
            vars = [vars]
        for s in vars:
            temp = re.sub(r'[^a-z0-9\s]','',s.strip().lower())
            temp = re.sub(r'\s+', '_', temp)
            clean_vars.append(temp)
        return clean_vars
    except Exception as e:
        logger.error("Error cleaning string : %s", e)
        raise         

def mandatory_column_check(df : pd.DataFrame, columns : list) -> bool:
    """
    Function to check if the given columnsa re present in the dataframe

    Args:
        df (pd.DataFrame): Dataframe to check for columns presence
        columns (list): Mandatory columns to check for

    Returns:
        bool: Boolean value representing status
    """
    try:
        if isinstance(columns, str):
            columns = [columns]
        missing_col = []
        df_headers_list = df.columns.to_list()
        mandatory_columns = clean_string(vars = columns)
        df_headers = clean_string(vars= df_headers_list)
        for col in mandatory_columns:
            if col.lower() not in [s.lower() for s in df_headers]:
                missing_col.append(col)
        if len(missing_col) >= 1:
            return False
        else:
            return True
    except Exception as e:
        logger.error("Error checking for mandatory columns : %s", e)
        raise

def drop_dataframe_columns(df : pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Function to drop columns form a dataframe

    Args:
        df (pd.DataFrame): Dataframe to manipulate
        columns (list): Columns to be dropped from the dataframe

    Returns:
        pd.DataFrame: Formatted fataframe
    """
    try:
        final_df = df.drop(columns=columns)
        return final_df
    except Exception as e:
        logger.error("Error droping columns : %s", e)
        raise

def dataframe_join(df1 :pd.DataFrame, df2 : pd.DataFrame,
                   join_column: str) -> pd.DataFrame:
    """ 
    Function to join different dataframes on a paticular column

    Args:
        df1 (pd.DataFrame): Left dataframe
        df2 (pd.DataFrame): Right dataframe
        join_column (str): Join column

    Returns:
        pd.DataFrame: Cross join of both dataframes
    """
    try:
        formatted_join_col = clean_string(join_column)[0]
        df_list = [df1, df2]
        for i in range(len(df_list)): # pylint: disable=consider-using-enumerate
            for col in df_list[i]:
                formatted_col_name = clean_string(col)
                df_list[i].rename(columns={col: formatted_col_name[0]}, inplace=True)
        merged_df = pd.merge(df1, df2, on=formatted_join_col, how ='inner')
        return merged_df
    except Exception as e:
        logger.error("Error merging dataframes : %s", e)
        raise

def marks_custom_encoder(df : pd.DataFrame) -> pd.DataFrame:
    """
    Function to implement custom one hot encoding for categorical variables

    Args:
        df (pd.DataFrame): Dataframe to manipulate

    Returns:
        pd.DataFrame: Updated dataframe with one hot encoded columns
    """
    try:
        barre_chords_temp = ["F", "F♯", "G♯", "A♯", "B","Fm", "F♯m", "Gm", "G♯m","A♯m",
                            "Bm","Bb", "Cm", "C♯m", "D♯m","F7", "F♯7", "G♯7", "A♯7", 
                            "C♯7", "D♯7","Fm7","F♯m7", "Gm7", "G♯m7", "Am7", "A♯m7",
                            "Bm7", "Cm7", "C♯m7", "Dm7", "D♯m7", "E7"]
        major_minor_chords_temp = ["A", "C", "D", "E", "G", "Am", "Dm", "Em"]
        special_chords = []
        df['formatted_chords'] = df['chords'].apply(lambda x:
            re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', x)).strip())
        all_chords = set(chord for row in df['formatted_chords'] for chord in row.split()) # pylint: disable=redefined-outer-name
        barre_chords = [x.lower() for x in barre_chords_temp]
        barre_chords = [x for x in barre_chords if x not in major_minor_chords_temp]
        major_minor_chords = [x.lower() for x in major_minor_chords_temp]
        for chord in all_chords:
            if chord.lower() not in barre_chords and chord.lower() not in major_minor_chords:
                special_chords.append(chord)

        df['song_length_temp'] = df['formatted_chords'].apply(lambda x: len(x)) # pylint: disable=unnecessary-lambda
        df['song_length'] = (((df['song_length_temp'] - df['song_length_temp'].min())/
                              (df['song_length_temp'].max()-df['song_length_temp'].min()))) * 15
        df['distinct_chords'] = df["formatted_chords"].apply(lambda x: set(x.split()))
        df['barre_chords_metric'] = (df['distinct_chords'].apply(lambda x:
            sum(chord in barre_chords_temp for chord in x)) * 2)
        df['major_minor_chords_metric'] = df['distinct_chords'].apply(lambda x:
            sum(chord in major_minor_chords_temp for chord in x))
        df['special_chords'] = (df['distinct_chords'].apply(lambda x:
            sum(chord in special_chords for chord in x)) * 3)
        return df
    except Exception as e:
        logger.error("Error one hot encoding data : %s", e)
        raise

def exercise_custom_encoder(df: pd.DataFrame)-> pd.DataFrame :
    """
    Function to encode features in the exercise dataframe

    Args:
        df (pd.DataFrame): Dataframe to manipulate

    Returns:
        pd.DataFrame: Resultant dtaframe
    """
    try:
        barre_chords_temp = ["F", "F♯", "G♯", "A♯", "B","Fm", "F♯m", "Gm", "G♯m","A♯m",
                            "Bm","Bb", "Cm", "C♯m", "D♯m","F7", "F♯7", "G♯7", "A♯7", 
                            "C♯7", "D♯7","Fm7","F♯m7", "Gm7", "G♯m7", "Am7", "A♯m7",
                            "Bm7", "Cm7", "C♯m7", "Dm7", "D♯m7", "E7"]
        major_minor_chords_temp = ["A", "C", "D", "E", "G", "Am", "Dm", "Em"]
        special_chords = []
        df_exploded = df['chord_progression'].str.split(',').explode()
        all_chords = df_exploded.unique().tolist() # pylint: disable=redefined-outer-name
        barre_chords = [x.lower() for x in barre_chords_temp]
        barre_chords = [x for x in barre_chords if x not in major_minor_chords_temp]
        major_minor_chords = [x.lower() for x in major_minor_chords_temp]
        for chord in all_chords:
            if chord.lower() not in barre_chords and chord.lower() not in major_minor_chords:
                special_chords.append(chord)    
        df['barre_chords_metric'] = (df['chord_progression'].apply(lambda x:
        sum(chord in barre_chords_temp for chord in x)) * 2)
        df['major_minor_chords_metric'] = df['chord_progression'].apply(lambda x:
            sum(chord in major_minor_chords_temp for chord in x))
        df['special_chords'] = (df['chord_progression'].apply(lambda x:
            sum(chord in special_chords for chord in x)) * 3)
        df['tempo_mattric'] = ((df['tempo'] - 40) / (200 - 40))
        return df
    except Exception as e:
        logger.error("Error encoding exercise data : %s", e)
        raise

def get_universal_chords (df: pd.DataFrame) -> list:
    """
    Function to get a list of all chords

    Args:
        df (pd.DataFrame): Dtaframe to extract chords cfrom
        columns (list): Chord column name

    Returns:
        list: List of all unqieu chords
    """
    try:
        df['formatted_chords'] = df['chords'].apply(lambda x:
            re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', x)).strip())
        all_chords_list = set(chord for row in df['formatted_chords'] for chord in row.split())
        return all_chords_list
    except Exception as e:
        logger.error("Error getting universal chords list : %s", e)
        raise

def chords_to_vector(chord_list, universal_chords):
    """
    Convert list of chords into a binary vector based on universal chords.
    """
    try:
        return [1 if chord in chord_list else 0 for chord in universal_chords]
    except Exception as e:
        logger.error("Error in coverting chord to vector : 5s", e)
        raise
    
def scaler_function(df:pd.DataFrame, columns:list)-> pd.DataFrame:
    """
    Function to create and scale feature vectors

    Args:
        df (pd.DataFrame): Dataframe to manipulate
        columns (list): Columns to engineer

    Returns:
        pd.DataFrame: Resultant Dataframe
    """
    try:
        if isinstance(columns, str):
            columns = [columns]
        scaler = MinMaxScaler()
        chords_scaled = scaler.fit_transform(df[columns])
        df_scaled = pd.DataFrame(chords_scaled,columns=[c + "_scaled" for c in columns])
        df = pd.concat([df.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
        return df
    except Exception as e:
        logger.error("Error in scaling columns: 5s", e)
        raise

def create_feature_vector(df:pd.DataFrame, columns:list)-> pd.DataFrame:
    """
    Function to create final feature vector

    Args:
        df (pd.DataFrame): Dataframe to manipulate
        columns (list): _descriColumns to engineerption_

    Returns:
        pd.DataFrame: Resultant Dataframe
    """
    try:
        df['feature_vector'] = df.apply(
        lambda row: row['chord_vector'] + [row[col] for col in columns],
        axis=1)
        return df
    except Exception as e:
        logger.error("Error in creating feature vectors: 5s", e)
        raise

def exercise_build_vector(row): # pylint: disable=missing-function-docstring
    try:        
        feature_cols = ['barre_chords_metric_scaled','major_minor_chords_metric_scaled',
                        'special_chords_scaled','tempo_mattric_scaled']
        chord_vec = list(row['chord_vector'])
        extra = [float(row[col]) for col in feature_cols]
        return chord_vec + extra
    except Exception as e:
        logger.error("Error in building feature vector for exercise df: 5s", e)
        raise

def marks_build_vector(row): # pylint: disable=missing-function-docstring
    try:
        feature_cols = ['barre_chords_metric_scaled','major_minor_chords_metric_scaled',
                        'special_chords_scaled','song_length_scaled']
        chord_vec = list(row['chord_vector'])
        extra = [float(row[col]) for col in feature_cols]
        return chord_vec + extra
    except Exception as e:
        logger.error("Error in c\building feature vector for marks df: 5s", e)
        raise


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    song_data_path = os.path.join(base_dir, '..','data','raw','songs_data.parquet')
    spotify_data_path = os.path.join(base_dir,'..','data','raw','spotify_tracks.parquet')
    exercise_data_path = os.path.join(base_dir, '..','data','raw','chord_exercises.csv')

    song_data = pd.read_parquet(song_data_path)
    spotify_data = pd.read_parquet(spotify_data_path)
    exercise_data = pd.read_csv(exercise_data_path)
    marks_data = dataframe_join(song_data, spotify_data, join_column='spotify_song_id')
    marks_data = drop_dataframe_columns(marks_data, columns=['id','releasedate','decade',
                                                             'rockgenre','artistid',
                                                             'spotifysongid','spotifyartistid'])
    marks_file_path = os.path.join(base_dir, '..', 'data', 'raw', 'marks_data.parquet')
    marks_data_ohe = marks_custom_encoder(df=marks_data)
    exercise_data_ohe = exercise_custom_encoder(exercise_data)
    all_chords = get_universal_chords(marks_data_ohe)
    marks_data_ohe['chord_vector'] = marks_data_ohe['distinct_chords'].apply(
        lambda x: chords_to_vector(x, all_chords))
    exercise_data_ohe['chord_vector'] = exercise_data_ohe['chord_progression'].apply(
        lambda x: chords_to_vector(x.split(','), all_chords)
    )
    exercise_data_final = scaler_function(
        df = exercise_data_ohe, columns=['barre_chords_metric', 
                                         'major_minor_chords_metric', 
                                         'special_chords','tempo_mattric'])
    exercise_data_final['feature_vector'] = exercise_data_final.apply(exercise_build_vector, axis=1)
    marks_data_final = scaler_function(df = marks_data_ohe, columns=
                                       ['barre_chords_metric', 'major_minor_chords_metric', 
                                        'special_chords','song_length'])
    marks_data_final['feature_vector'] = marks_data_final.apply(marks_build_vector, axis=1)
    
    marks_data_final.to_parquet(os.path.join(
        base_dir, '..', 'data', 'processed', 'marks_data.parquet'))
    exercise_data_ohe_path = os.path.join(
        base_dir, '..','data','processed','chord_exercises.parquet')
    exercise_data_final.to_parquet(exercise_data_ohe_path)
