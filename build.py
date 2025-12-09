import faiss
from utils import *
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import json

HOME     = Path.cwd().parent
DATA_DIR = HOME / 'rag_project' / 'scikit-learn-docs' / '_sources' / 'modules'
DB_DIR   = HOME / 'rag_project' / 'vector_store'

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

all_chunks = []
all_embeddings = []

# Processing files and creating embeddings
for file in DATA_DIR.glob('*.txt'):
    print(f"Processing {file.name}")
    chunks = chunk_rst_text(file, file.name)

    for chunk in chunks:
        # Embed
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['text']
        )
        embedding = response.data[0].embedding

        all_chunks.append(chunk)
        all_embeddings.append(embedding)

# Convert to numpy
embeddings = np.array(all_embeddings, dtype='float32')
dim = embeddings.shape[1]

print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding dimension: {dim}")

# Create different index types
print("\nCreating indices...")

# 1. Flat (exact search)
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(embeddings)
faiss.write_index(index_flat, str(DB_DIR / 'IndexFlatL2.index'))

# 2. IVF (approximate, needs training)
nlist = 10  # Number of clusters (adjust based on data size)
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(embeddings)  # IVF requires training!
index_ivf.add(embeddings)
faiss.write_index(index_ivf, str(DB_DIR / 'IndexIVFFlat.index'))

# 3. HNSW (fast approximate)
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # 32 = M (connections per layer)
index_hnsw.add(embeddings)
faiss.write_index(index_hnsw, str(DB_DIR / 'IndexHNSWFlat.index'))

# Save chunks metadata
with open(DB_DIR / 'chunks_metadata.json', 'w') as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nSaved all indices to {DB_DIR}")
