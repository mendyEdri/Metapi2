## Metapi2 Demo

This repository contains a small Streamlit application for experimenting with
chunking and clustering text using OpenAI embeddings. The app can now also
compute a **weight** for each chunk based on its cosine similarity to the
average embedding. The weights are softmax‑normalized so they sum to one and
allow you to see which parts of your prompt are most influential. The UI shows
chunks ordered by weight and assembles a weighted prompt accordingly.

Beyond weighting, the helper functions in :mod:`clustering` let you build a
similarity graph between chunks and rank them using classic centrality metrics
such as PageRank or betweenness centrality. This makes it easy to identify the
most connected or influential pieces of a prompt.

By default the app attempts to parse the provided prompt as XML and clusters
the textual content of the XML elements. If the input isn't valid XML it falls
back to a plain text splitting strategy using LangChain's
``RecursiveCharacterTextSplitter``.
