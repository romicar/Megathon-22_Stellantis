import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.express as px
from collections import defaultdict
from scipy.spatial.distance import cdist
import difflib
import sys
#%matplotlib inline
def preprocess():
    global spotify_data
    spotify_data = pd.read_csv('./data/data.csv.zip')
    genre_data = pd.read_csv('./data/data_by_genres.csv')
    data_by_year = pd.read_csv('./data/data_by_year.csv')

    sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
#fig = px.line(data_by_year, x='year', y=sound_features)
#fig.show()

#fig = px.line(data_by_year, x='year', y='tempo')
#fig.show()
#top10_genres = genre_data.nlargest(10, 'popularity')
#fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
#fig.show()

    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans( n_clusters=10))])
    X = genre_data.select_dtypes(np.number)
    cluster_pipeline.fit(X)
    genre_data['cluster'] = cluster_pipeline.predict(X)

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
# fig.show()

    global song_cluster_pipeline
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=2))],verbose=True)
    X = spotify_data.select_dtypes(np.number)
    global number_cols
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    spotify_data['cluster_label'] = song_cluster_labels

    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = spotify_data['name']
    projection['cluster'] = spotify_data['cluster_label']

    fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
# fig.show()



    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    
    return spotify_data

def get_song_data(song, spotify_data):

    """
    Gets the song data for a specific song. The song argument takes the form of a dictionary with
    key-value pairs for the name and release year of the song.
    """

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):

    """
    Gets the mean vector for a list of songs.
    """

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):

    """
    Utility function for flattening a list of dictionaries.
    """

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):

    """
    Recommends songs based on a list of previous songs that a user has listened to.
    """

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

#answer = recommend_songs([{'name': 'Night Changes', 'year': 2014}, {'name': 'Best Song Ever', 'year': 2013}], spotify_data,100)
with open('log', 'w') as sys.stdout:
    preprocess()
    #answer = recommend_songs([{'name': 'Ghost Of You', 'year': 2018},{'name': 'Someone To Stay', 'year': 2017},{'name': 'Acquainted','year': 2015},
    #{'name': 'All We Know','year': 2016},{'name': 'Let Me','year': 2017}],spotify_data,20)
    answer = recommend_songs([{'name': 'Johnny B. Goode','year': 1959},{'name': 'Starman - 2012 Remaster','year':1972},{'name': 'Killer Queen','year' :1974}],
        spotify_data,20)
sys.stdout= sys.__stdout__
    
for i in answer:
    print(i)
