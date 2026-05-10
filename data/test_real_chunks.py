from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from script import Resume_Chunks

contents = [
    f"{chunk['Section']}: {chunk['Content']}"
    for chunk in Resume_Chunks
]


print("Total chunks:", len(contents))
print("First chunk:", contents[0])

model = SentenceTransformer("all-MiniLM-L6-v2")
content_encode= model.encode(contents)

index = faiss.IndexFlatL2(384)
index.add(content_encode)

print("Vectors in index:", index.ntotal)

questions = [
    "How many years of experience does he have?",
    "What tools and technologies does he know?",
    "Has he worked in a team environment?",
    "What is his educational background?",
    "Is he open to relocation?",
    "What salary is he expecting?",
    "Does he need visa sponsorship?",
    "What kind of roles is he looking for?",
    "Has he led any projects or teams?",
    "What are his most impressive projects?"
]

for question in questions:
    question_vector = model.encode(question).reshape(1, -1)
    distances, indices = index.search(question_vector, k=3)

    print(f"\nQ: {question}")
    print("Top 3 chunks:")
    for rank, i in enumerate(indices[0]):
        chunk = Resume_Chunks[i]
        dist = distances[0][rank]
        print(f"  {rank + 1}. [{chunk['Section']}] dist={dist:.4f}")



