## Simple RAG for text summarization

# How to run the scripts

1. clone the repo

```
git clone (link of the repo)
```

2. install dependences

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

3. run scripts

```
sims.py: python sims.py "Meaning of life" --metric cosine --file "moby-dick.txt"
```

- try out youre own txt files. Just upload them to the base directory "./" of the repo
- added different similarity metrics (cosine, dot, euclidean, manhattan) to demonstrate the differences

# Similarities

1. The **dot product** is used to find the angle between two vectors and to project one vector onto another, it provides a way to measure the alignment between to vectors.

```
a * b = $a_1$ * $b_1$ + $a_n$ * $b_n$
```
