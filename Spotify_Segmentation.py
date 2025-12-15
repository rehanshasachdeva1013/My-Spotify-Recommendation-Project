#!/usr/bin/env python
# coding: utf-8

# # Spotify Songs’ Genre Segmentation
# 
# ## 1. Initialization and Data Loading

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set plot style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load dataset
try:
    df = pd.read_csv('spotify dataset.csv')
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    display(df.head())
except FileNotFoundError:
    print("Error: 'spotify dataset.csv' not found. Please ensure the file is in the same directory.")


# ## 2. Data Pre-processing
# - Checking for missing values
# - Data cleaning and formatting

# In[ ]:


# Check info and missing values
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing values (if any significant amount, or fill them)
df = df.dropna()
print(f"\nShape after dropping missing values: {df.shape}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
df = df.drop_duplicates()
print(f"Shape after dropping duplicates: {df.shape}")


# ## 3. Exploratory Data Analysis (EDA)
# - Visualizing feature distributions
# - Correlation matrix
# - Genre-based analysis

# In[ ]:


# Feature columns for analysis
feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df[feature_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Audio Features')
plt.show()


# In[ ]:


# Distribution of Danceability and Energy
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(df['danceability'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Danceability')

sns.histplot(df['energy'], kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribution of Energy')

plt.show()


# In[ ]:


# Top 10 Genres by Popularity (if 'track_popularity' exists, otherwise just count)
if 'playlist_genre' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y='playlist_genre', data=df, order=df['playlist_genre'].value_counts().index, palette='viridis')
    plt.title('Count of Songs by Genre')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.show()


# ## 4. Clustering
# - Feature Scaling
# - K-Means Clustering
# - PCA for Visualization

# In[ ]:


# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# Determining optimal k using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


# Applying K-Means with an optimal k (e.g., k=5 or chosen from elbow plot)
# Let's assume k=6 based on typical genre counts, or user can adjust
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
print(df['cluster'].value_counts())


# In[ ]:


# Visualizing Clusters using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['cluster'] = clusters
pca_df['genre'] = df['playlist_genre'] if 'playlist_genre' in df.columns else 'Unknown'

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='tab10', alpha=0.6)
plt.title('Clusters Visualization (PCA)')
plt.show()


# ## 5. Recommendation System
# Build a simple recommender based on clusters.

# In[ ]:


def recommend_songs(song_name, df, num_recommendations=5):
    # Check if song exists
    # Note: Dataset might have 'track_name' column. Adjust if different.
    if 'track_name' not in df.columns:
         return "Error: 'track_name' column not found in dataset."
    
    song_row = df[df['track_name'].str.lower() == song_name.lower()]
    
    if song_row.empty:
        return f"Song '{song_name}' not found in the dataset."
    
    # Get cluster of the song
    song_cluster = song_row.iloc[0]['cluster']
    
    # Filter songs from the same cluster
    cluster_songs = df[df['cluster'] == song_cluster]
    
    # Randomly sample recommendations (or could be based on distance)
    recommendations = cluster_songs.sample(min(num_recommendations, len(cluster_songs)))
    
    return recommendations[['track_name', 'track_artist', 'playlist_genre']]

# Test Recommendation
# Pick a song from the dataframe head to test
sample_song = df['track_name'].iloc[0]
print(f"Recommendations for '{sample_song}':")
print(recommend_songs(sample_song, df))

