## Simple RAG for text summarization

# How to run the scripts

1. clone the repo

```
git clone https://github.com/makarwu/RAGsum.git
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
a * b = a_1 * b_1 + a_n * b_n
```

2. **Cosine similarity** divides the angle of the vectors by the length (magnitude) of the vectors

```
cosine_similarity(A, B) = (A dot B) / (|A| * |B|)
```

3. **Euclidean Distance** measures the straight-line distance between two points.

```
d(P,Q) = sqrt((x2 - x1)^2 + (y2 - y1)^2 + ... + (zn - z1)^2)
```

4. **Manhattan Distance** demonstrates the distance between two points in a grid-based system.

```
d = |x2 - x1| + |y2 - y1| + ... + |wn - wn|
```
