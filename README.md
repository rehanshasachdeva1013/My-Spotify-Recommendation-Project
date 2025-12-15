# 🎵 Spotify Genre Segmentation & Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> **Discover the hidden patterns in music!** This project analyzes Spotify song data to uncover genre clusters and builds an intelligent recommendation engine powered by Machine Learning.

---

## 🚀 Overview

Have you ever wondered why Spotify's "Discover Weekly" is so good? It's all about data! 
This project takes a deep dive into audio features like **danceability**, **energy**, **acousticness**, and **tempo** to categorize songs automatically.

We use **K-Means Clustering** to segregate songs into distinct groups and then recommend similar tracks based on mathematical proximity.

## ✨ Key Features

*   **📊 Exploratory Data Analysis (EDA):** Visualizing the correlation between audio features and genres.
*   **🤖 Unsupervised Learning:** Using K-Means Clustering to group 30,000+ songs.
*   **📉 Dimensionality Reduction:** Visualizing high-dimensional data in 2D using PCA (Principal Component Analysis).
*   **🎧 Content-Based Recommendation:** enter a song, get 5 curated recommendations from the same cluster.

## 🛠️ Tech Stack

*   **Core:** Python 3
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (KMeans, PCA)

## 📸 Snapshots

### Correlation Heatmap
*Understanding how features interact with each other.*

### Cluster Visualization (PCA)
*Seeing the genre separation in 2D space.*

## 🏁 Getting Started

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/agastyajxa/Spotify-Recommendation-Project.git
    cd Spotify-Recommendation-Project
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis**
    Launch Jupyter Notebook and open `Spotify_Segmentation.ipynb`:
    ```bash
    jupyter notebook
    ```

## 🧠 How it Works

1.  **Data Loading:** We load the `spotify dataset.csv`.
2.  **Preprocessing:** Clean missing values and normalize the data using `StandardScaler`.
3.  **Clustering:** The **Elbow Method** determines the optimal number of clusters (k).
4.  **Prediction:** The model assigns a cluster ID to every song.
5.  **Recommendation:** When you query a song, the system looks up its cluster and samples other songs from the same "neighborhood".

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/agastyajxa/Spotify-Recommendation-Project/issues).

---

Made with ❤️ and 🎧 by **Agastya**
