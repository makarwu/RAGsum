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