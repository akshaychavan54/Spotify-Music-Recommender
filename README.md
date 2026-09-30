# 🎵 Spotify Music Recommended System

A Streamlit-based music recommendation app built using Unsupervised Machine Learning. The system clusters tracks using K-Means and PCA based on audio features to deliver personalized song recommendations.

---

## 🚀 Features
- Data Scaling: Uses StandardScaler to normalize audio features (danceability, energy, tempo, loudness, valence, etc.).
- Optimal Clustering: K-Means clustering algorithm (K=7) optimized via the Elbow Method.
- Dimensionality Reduction: Implements Principal Component Analysis (PCA) for cluster visualization.
- Interactive UI: Dynamic web application built with Streamlit.

---

## 🛠️ Tech Stack
- Language: Python
- Machine Learning: Scikit-Learn (KMeans, StandardScaler, PCA)
- Data Manipulation: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Web Framework: Streamlit

---

## 📁 Repository Structure
```text
├── app.py                     # Main Streamlit web application
├── Spotify_recommended.csv    # Processed audio feature dataset
├── Spotify.pkl                # Pre-trained K-Means & PCA model
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation