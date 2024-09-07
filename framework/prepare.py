"""
makarwu
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### 1. DATA PREPARATION ###

chunks = []
with open('../the-verdict.txt', 'r') as file:
    chunk = []
    for line in file:
        if line.strip() == "":
            if chunk:
                chunks.append("\n".join(chunk))
                chunk = []
        else:
            chunk.append(line.strip())

chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}

### 2. RETRIEVAL SYSTEM ###

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chunks)

def retrieve_chunks(query, k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_k_indices]

retrieved_chunks = retrieve_chunks("love and beauty")
for chunk, score in retrieved_chunks:
    print(f"Score: {score:.2f}\n{chunk}\n")