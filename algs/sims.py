"""
makarwu
"""
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm

def load_chunks(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    chunks = []
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            if line.strip() == "":
                if chunk:
                    chunks.append("\n".join(chunk))
                    chunk = []
            else:
                chunk.append(line.strip())

    return chunks

def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1, vec2).flatten()

def dot_product(vec1, vec2):
    return np.dot(vec1.toarray(), vec2.toarray().T).flatten()

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, axis=1)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2), axis=1)

def retrieve_chunks(query, vectorizer, X, chunks, k=5, metric='cosine'):
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
    parser.add_argument('--file', type=str, required=True, choices=['moby-dick.txt', 'peter-pan.txt', 'the-verdict.txt'], 
                        help="The text file to analyze.")
    
    args = parser.parse_args()

    file_path = f"../{args.file}"
    chunks = load_chunks(file_path)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks)
    retrieved_chunks = retrieve_chunks(args.query, vectorizer, X, chunks, metric=args.metric)
    
    for chunk, score in retrieved_chunks:
        print(f"Score: {score:.2f}\n{chunk}\n")

if __name__ == "__main__":
    main()
