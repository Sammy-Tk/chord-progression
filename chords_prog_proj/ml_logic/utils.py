import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


'''
GET CHORD COUNT & OPTIONAL DISTRIBUTION
'''
def count_chords(file_name, low_freq_to_remove=10, histplot=False, ascending=False, out_of_pre=False):

    if out_of_pre == True:
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(root_path, 'mlops/data/processed', file_name)
        df = pd.read_csv(data_path)
    else:
        df = file_name.copy()

    chords_count_dict = {}
    for song in df['chords']:
        if type(song) == str:
            song = ast.literal_eval(song)
        song_dict = dict(Counter(song))
        for chord, count in song_dict.items():
            if chord in chords_count_dict:
                chords_count_dict[chord] = chords_count_dict[chord] + count
            else:
                chords_count_dict[chord] = count

    slim_chord_counts_dict = {}
    for chord, count in chords_count_dict.items():
        if count <= low_freq_to_remove:
            pass
        else:
            slim_chord_counts_dict[chord] = count

    chord_count_df = pd.Series(slim_chord_counts_dict).to_frame('chord_count')
    chord_count_df.sort_values(by='chord_count', ascending=ascending, inplace=True)

    if histplot == True:
        top_15_chords_df = chord_count_df.head(15)
        sns.set_theme(style="whitegrid")
        chords_fig = sns.barplot(x=top_15_chords_df.index,
                                 y=top_15_chords_df.chord_count,
                                 palette='gist_ncar')
        chords_fig.set_xlabel('Chord')
        chords_fig.set_ylabel('Number of Appearances')
        chords_fig.set_title('Most Common Chords')
        ylabels = [f'{y:,}'[:7] for y in chords_fig.get_yticks()]
        chords_fig.set_yticklabels(ylabels)
        plt.show()

    return chord_count_df


'''
GET GENRE DISTRIBUTION
'''
def count_genres(final_df, histplot=False):

    genre_count_ser = final_df['genres'].value_counts()

    genre_count_df = genre_count_ser.to_frame('genre_count')

    genre_count_df.sort_values(by='genre_count', ascending=False, inplace=True)

    if histplot == True:
        genres_fig = sns.histplot(genre_count_df, bins=100)
        genres_fig.set(xticklabels=[])
        genres_fig.set_xlabel('genres')
        plt.show()
    else:
        pass

    return genre_count_df


'''
GET ARTIST DISTRIBUTION
'''
def count_artists(final_df, histplot=False):

    artist_count_ser = final_df['artist_name'].value_counts()

    artist_count_df = artist_count_ser.to_frame('artist_count')

    artist_count_df.sort_values(by='artist_count', ascending=False, inplace=True)

    if histplot == True:
        artists_fig = sns.histplot(final_df['artist_name'], bins=100)
        artists_fig.set(xticklabels=[])
        artists_fig.set_xlabel('artists')
        plt.show()
    return artist_count_df

#EC
