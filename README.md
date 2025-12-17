# Song Recommendation System

## Project Overview
This project is a **Content-Based Recommender System**. Unlike traditional systems that look at what other users like, this system looks at the **lyrics** of the songs to find similarities.

## How it Works
1. **Data Collection:** Uses the Spotify Million Song Dataset.
2. **Text Preprocessing:** Cleans the lyrics by removing stop words.
3. **Vectorization:** Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical data.
4. **Similarity Scoring:** Uses **Cosine Similarity** to calculate the distance between song vectors and find the closest matches.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn
* **Environment:** Google Colab

## How to Run
1. Upload the `spotify_millsongdata.csv` to the environment.
2. Run the notebook cells sequentially.
3. Use the `recommend("Song Name")` function to get results.
