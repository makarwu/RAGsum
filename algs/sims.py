"""
makarwu
"""
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm

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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chunks)

def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1, vec2).flatten()

def dot_product(vec1, vec2):
    return np.dot(vec1.toarray(), vec2.toarray().T).flatten()

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, axis=1)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2), axis=1)

def retrieve_chunks(query, k=5, metric='cosine'):
    query_vec = vectorizer.transform([query])

    if metric == 'cosine':
        similarities = cosine_sim(query_vec, X)
    elif metric == 'dot':
        similarities = dot_product(query_vec, X)
    elif metric == 'euclidean':
        similarities = euclidean_distance(query_vec.toarray(), X.toarray())
    elif metric == 'manhattan':
        similarities = manhattan_distance(query_vec.toarray(), X.toarray())
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_k_indices]
   

def main():
    parser = argparse.ArgumentParser(description="Retrieve text chunks using various similarity or distance measures.")
    parser.add_argument('query', type=str, help="Youre query text.")
    parser.add_argument('--metric', type=str, required=True, choices=['cosine', 'dot', 'euclidean', 'manhattan'], 
                        help="The similarity/distance metric to use.")
    args = parser.parse_args()

    retrieved_chunks = retrieve_chunks(args.query, metric=args.metric)
    for chunk, score in retrieved_chunks:
        print(f"Score: {score:.2f}\n{chunk}\n")

if __name__ == "__main__":
    main()
