from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load Data (We take a sample of 5k songs to run fast on free server)
df = pd.read_csv('spotify_small.csv').sample(5000).reset_index(drop=True)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['text'].fillna(''))
similarity = cosine_similarity(matrix)

@app.get("/")
def home():
    return {"message": "Music Recommendation API is Live"}

@app.get("/recommend")
def recommend(song: str):
    song = song.strip()
    # Find the song (case insensitive)
    matches = df[df['song'].str.lower() == song.lower()]
    
    if matches.empty:
        return {"error": "Song not found in sample. Try another."}
    
    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Return top 5
    recs = []
    for i in sorted_scores[1:6]:
        recs.append(df.iloc[i[0]]['song'])
        

    return {"input_song": song, "recommendations": recs}
