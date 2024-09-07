"""
makarwu
"""
import re
from transformers import pipeline
from prepare import retrieve_chunks

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

retrieved_chunks = retrieve_chunks("love and beauty")
for chunk, _ in retrieved_chunks:
    summary = summarizer(chunk, max_length=50, min_length=25, do_sample=True)
    print("Summary:", summary[0]['summary_text'])

if __name__ == "main":
    def rag_summarization(query):
        retrieved_chunks = retrieve_chunks(query)
        all_summaries = []
        for chunk, score in retrieved_chunks:
            summary = summarizer(chunk, max_length=50, min_length=25, do_sample=True)
            all_summaries.append(summary[0]['summary_text'])
        
        return " ".join(all_summaries)

    print(rag_summarization("Meaning of life"))
