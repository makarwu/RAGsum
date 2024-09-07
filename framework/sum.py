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


