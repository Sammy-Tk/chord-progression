import pandas as pd
import os
import time
from colorama import Fore, Style

from chords_prog_proj.ml_logic.data import *
from chords_prog_proj.ml_logic.utils import *


def pre_clean():
    """Read raw data, pre-clean and concatenate"""

    # Get raw data
    df_kaggle_raw = get_data_kaggle()
    print(Fore.GREEN + f'\n(Kaggle dataset): Raw DataFrame with shape {df_kaggle_raw.shape}' + Style.RESET_ALL)
    df_lstm_realbook_raw = get_data_lstm_realbook()
    print(Fore.GREEN + f'\n(LSTM Realbook dataset): Raw DataFrame with shape {df_lstm_realbook_raw.shape}' + Style.RESET_ALL)

    # Remove duplicate songs and select columns
    df_kaggle_selected_cols = drop_cols(remove_duplicates(df_kaggle_raw))
    print(Fore.GREEN + f'\n(Kaggle dataset) Removed duplicate songs and selected columns: DataFrame with shape {df_kaggle_selected_cols.shape}' + Style.RESET_ALL)
    # Select columns
    df_lstm_realbook_selected_cols = expand_cols(df_lstm_realbook_raw)
    print(Fore.GREEN + f'\n(LSTM Realbook dataset) Selected columns: DataFrame with shape {df_lstm_realbook_selected_cols.shape}' + Style.RESET_ALL)

    # Concatenate DataFrames
    df_concatenated = pd.concat(
        [df_kaggle_selected_cols, df_lstm_realbook_selected_cols],
        ignore_index=True
        )

    print(Fore.GREEN + f'\nData read and merged. Total: {len(df_concatenated):,} songs.' + Style.RESET_ALL)

    print(Fore.GREEN + f'\n✅ Pre-clean finished.' + Style.RESET_ALL)

    return df_concatenated


def clean(df_concatenated):
    df_cleaned = df_concatenated.copy()

    # Remove songs that contain guitar tabs
    df_cleaned = remove_guitar_tabs(df_cleaned)
    print(Fore.GREEN + f'\nSongs with guitar tabs removed. Total: {len(df_cleaned):,} songs.' + Style.RESET_ALL)

    cleaned_chords_column = clean_chords(df_cleaned['chords'])
    df_cleaned['chords'] = cleaned_chords_column

    print(f'\n✅ Chords cleaned.')

    return df_cleaned


def preprocess(get_distributions=False):

    print(Fore.BLUE + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)
    df_concatenated = pre_clean()

    print(Fore.BLUE + f'\n✅ Cleaning chords... may take up to 1 minute.' + Style.RESET_ALL)
    cleaned_df = clean(df_concatenated)

    # drop duplicates based on chords and song name
    new_columns_df = new_columns(cleaned_df)

    unreplicated_df = \
        new_columns_df.drop_duplicates(subset=['chords_list', 'song_name'],
                                    keep = 'last').reset_index(drop = True)

    # drop unnecessary columns
    unreplicated_df.drop(columns=['song_name', 'chords_list'], inplace=True)

    print(f'\n✅ Genre cleaning... may take up to 1 minute.')

    # clean genres
    slim_genres = genre_cleaning(unreplicated_df['genres'])
    clean_genres_df = unreplicated_df.copy()
    clean_genres_df['genres'] = slim_genres

    # get song length column, and filter by length
    song_len_df = song_length(clean_genres_df)
    final_df = filter_length(song_len_df, 8)

    print(f'{len(final_df)} songs kept during preprocessing.')

    if get_distributions == True:
        print(f'\n✅ Top five chords, genres, and artists being generated.')
        chord_count_df = count_chords(final_df, low_freq_to_remove=10,
                                      histplot=True, ascending=False)
        print(chord_count_df.head(5))
        genre_count_df = count_genres(final_df, histplot=True)
        print(genre_count_df.head(5))
        artists_df = count_artists(final_df, histplot=True)
        print(artists_df.head(5))

    # Save as CSV file
    # Generate timestamp
    ts = time.strftime("%d-%m-%y_%H:%M")

    # Construct the file path
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_proc_folder = os.path.join(root_path, LOCAL_DATA_PATH, 'processed')
    my_path = os.path.join(data_proc_folder, f"processed_data_{ts}.csv")

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(my_path), exist_ok=True)

    # Save the DataFrame to CSV
    final_df.to_csv(my_path, index=False)  # Add index=False to avoid including the DataFrame index
    print(f"✅ CSV saved at: {my_path}")

    print(f'\n✅ Preprocessor finished.')

    return None
