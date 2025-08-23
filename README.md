## Metapi2 Demo

This repository contains a small Streamlit application for experimenting with
chunking and clustering text using OpenAI embeddings. The app can now also
compute a **weight** for each chunk based on its cosine similarity to the
average embedding. The weights are softmax‑normalized so they sum to one and
allow you to see which parts of your prompt are most influential. The UI shows
chunks ordered by weight and assembles a weighted prompt accordingly.