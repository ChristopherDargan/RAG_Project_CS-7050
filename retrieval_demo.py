import faiss
import numpy as np
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import time

HOME = Path.cwd().parent
DB_DIR = HOME / 'rag_project' / 'vector_store'

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # if you're running locally, you likely cant without
                                                        # openai apikey for the embedding model for the queries
                                                        # if you do have one you can create a .env with the key inside


def load_index(index_name):
    index_path = DB_DIR / f'{index_name}.index'
    metadata_path = DB_DIR / 'chunks_metadata.json'

    index = faiss.read_index(str(index_path))

    with open(metadata_path, 'r') as f:
        chunks = json.load(f)

    return index, chunks


def search_docs(query_embedding, index, chunks, k=5):
    # Search and time it
    start = time.time()
    distances, indices = index.search(query_embedding, k)
    latency = (time.time() - start) * 1000  # Convert to ms

    # Format results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks[idx]
        results.append({
            'rank': i + 1,
            'id': chunk['id'],
            'header': chunk.get('header', 'N/A'),
            'filename': chunk['filename'],
            'distance': float(dist),
            'text': chunk['text'][:300]
        })

    return results, latency


# Load all indices
indices = {}
chunks = None

for index_name in ['IndexFlatL2', 'IndexIVFFlat', 'IndexHNSWFlat']:
    idx, chnks = load_index(index_name)
    indices[index_name] = idx
    if chunks is None:
        chunks = chnks  # Same chunks for all

print(f"Loaded {len(chunks)} chunks across 3 index types\n")

# Compare results side-by-side
# Interactive search
while True:
    query = input("\nSearch sklearn docs (or 'exit'): ")
    if query.lower() == 'exit':
        break

    # Embed query once
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array([response.data[0].embedding], dtype='float32')

    print(f"\n{'=' * 100}")
    print(f"Query: {query}")

    # Collect results from all indices
    all_results = {}
    for index_name, index in indices.items():
        results, latency = search_docs(query_embedding, index, chunks, k=5)
        all_results[index_name] = {'results': results, 'latency': latency}

    # Print comparison table
    print(f"\n{'Index Type':<20} {'Latency':<12} {'Top 3 Results'}")
    print('-' * 100)

    for index_name, data in all_results.items():
        results = data['results']
        latency = data['latency']
        top3 = ' | '.join([f"{r['header'][:20]}..." for r in results[:3]])
        print(f"{index_name:<20} {latency:>6.2f}ms    {top3}")

    # Print detailed results for each index
    print(f"\n{'=' * 100}")
    print("DETAILED RESULTS")
    print('=' * 100)

    for index_name, data in all_results.items():
        print(f"\n--- {index_name} ---")
        print(f"Latency: {data['latency']:.2f}ms\n")

        for r in data['results']:
            print(f"  Rank {r['rank']}: {r['id']}")
            print(f"    Distance: {r['distance']:.4f}")
            print(f"    Header: {r['header']}")
            print(f"    File: {r['filename']}")
            print(f"    Text: {r['text']}")
            print('=' * 100)
            print()