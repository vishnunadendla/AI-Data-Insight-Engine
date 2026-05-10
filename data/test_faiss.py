from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = [
    "Vishnu built Power BI dashboards at Epsilon",
    "Vishnu used Apache Airflow for ETL pipelines",
    "Vishnu studied at Rowan University"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

vector = model.encode(chunks)

print(vector)
print("shape:", vector.shape)

index = faiss.IndexFlatL2(384)
index.add(vector)
print("Vectors in index:", index.ntotal)

# Your question
question = "Does Vishnu know Power BI?"


question_vector = model.encode(question)

# Reshape for FAISS
# FAISS expects shape (1, 384) not (384,)
question_vector = question_vector.reshape(1, -1)

# Search index
# k=2 means "find 2 closest chunks"
distances, indices = index.search(
    question_vector, k=2
)

print("Closest chunks found:")
print("Indices:", indices)
print("Distances:", distances)

# Now print the ACTUAL chunk text
for i in indices[0]:
    print("→", chunks[i])
