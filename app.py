import streamlit as st
import pickle
import pandas as pd
import gdown

# --- Download and Load Files ---

# 1. Load the movie dataframe from pickle file
try:
    movies = pd.read_pickle('movies_df.pkl')
except FileNotFoundError:
    st.error("movies_df.pkl file not found. Please ensure it is uploaded to your GitHub repository.")
    st.stop()

# 2. Download the large similarity matrix from Google Drive
# This is the corrected and clean code for downloading the file.
google_drive_link = 'https://drive.google.com/file/d/1Mx5nfbasB2P8Ahxapbf1hhyKwdoMujKx/view?usp=sharing'
try:
    file_id = google_drive_link.split('/d/')[1].split('/view')[0]
except IndexError:
    st.error("Error: Could not extract file ID from your Google Drive link.")
    st.stop()

st.write("Downloading similarity data. This may take a moment...")
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'similarity_matrix.pkl', quiet=True)

# 3. Load the downloaded similarity matrix
try:
    with open('similarity_matrix.pkl', 'rb') as f:
        similarity = pickle.load(f)
except FileNotFoundError:
    st.error("The similarity matrix file was not found after downloading.")
    st.stop()


# --- Recommendation Logic ---

# Get the list of movie titles for the dropdown
movie_titles = sorted(list(movies['title'].unique()))

# Create a mapping from movie title to its integer index
movies['title_lower'] = movies['title'].str.lower()
title_to_index = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

def recommend(movie_title):
    movie_title = movie_title.lower()

    if movie_title not in title_to_index:
        return "Movie not found. Please check the spelling or try another title."

    movie_index = title_to_index[movie_title]
    distances = similarity[movie_index]
    movie_list = list(enumerate(distances))
    sorted_movie_list = sorted(movie_list, reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in sorted_movie_list[1:11]:
        recommended_movies.append(movies.iloc[i[0]]['title'])

    return recommended_movies


# --- Streamlit UI Code ---

st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_titles
)

if st.button('Show Recommendation'):
    recommendations = recommend(selected_movie_name)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)