import os
import pickle
import pandas as pd
import streamlit as st

# 1. Page Configuration (Title & Theme)
st.set_page_config(
    page_title="Spotify Music Recommender", page_icon="🎵", layout="centered"
)

# Custom Styling (Optional UI Enhancements)
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
    }
    stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #1DB954;
        color: white;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🎵 Spotify Music Recommender")
st.write("Select your favorite song to get recommendations!")

# 2. Data & Model Loading
DATASET_PATH = "Spotify_recommended.csv"
MODEL_PATH = "model.pkl"


@st.cache_data
def load_data():
  if os.path.exists(DATASET_PATH):
    return pd.read_csv(DATASET_PATH)
  return None


@st.cache_resource
def load_model():
  if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
      return pickle.load(f)
  return None


df = load_data()
model = load_model()


# 3. Recommendation Logic
def recommend_songs(song_name, num_recommendations=5):
  if df is None:
    return None, "❌ dataset.csv file not found!"

  matched = df[df["Song"].str.lower() == song_name.lower()]
  if matched.empty:
    return None, f"❌ Song '{song_name}' not found in the dataset."

  if "Clusters" in df.columns:
    song_cluster = matched.iloc[0]["Clusters"]
    cluster_songs = df[
        (df["Clusters"] == song_cluster)
        & (df["Song"].str.lower() != song_name.lower())
    ]
    recommendations = cluster_songs.sample(
        min(num_recommendations, len(cluster_songs))
    )[["Song", "Artist"]]
    return recommendations.to_dict(orient="records"), None
  elif model is not None:
    song_idx = matched.index[0]
    numeric_features = df.select_dtypes(include=["float64", "int64"])
    distances, indices = model.kneighbors(
        [numeric_features.iloc[song_idx]],
        n_neighbors=num_recommendations + 1,
    )
    rec_list = [
        {"Song": df.iloc[idx]["Song"], "Artist": df.iloc[idx]["Artist"]}
        for idx in indices[0][1:]
    ]
    return rec_list, None
  else:
    return None, "❌ Recommendation logic not available."


# 4. Streamlit UI Elements
if df is not None and "Song" in df.columns:
  song_list = sorted(df["Song"].dropna().unique())
  selected_song = st.selectbox(
      "Search or select a song from the dropdown:", song_list
  )

  if st.button("Get Recommendations 🚀"):
    with st.spinner("Fetching recommendations for you..."):
      recommendations, error = recommend_songs(selected_song)

    if error:
      st.error(error)
    elif recommendations:
      st.subheader(f"🎶 Recommended Songs for '{selected_song}':")
      for song in recommendations:
        with st.container():
          st.success(f"🎵 **{song['Song']}**  \n👤 Artist: *{song['Artist']}*")
else:
  st.error("⚠️ `dataset.csv` file not found. Please check your project folder.")

