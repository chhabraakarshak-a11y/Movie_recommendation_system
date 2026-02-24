import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

# Merge
movies = movies.merge(credits, on='title')

# Select features
movies = movies[['movie_id','title','overview','genres','keywords']]
movies.dropna(inplace=True)

# Combine text columns
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Save files
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model created successfully!")