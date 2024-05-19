import threading

from sentence_transformers import SentenceTransformer
import json
import pandas
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def get_vector(text):
    return model.encode(text).tolist()


def document_to_vector(input_file, output_file):
    with open(input_file) as f:
        documents = json.load(f)
    vectors = []
    for document in documents:
        vectors.append({"document": document, "vector": get_vector(document)})
        print(len(vectors))

    with open(output_file, 'w') as f:
        json.dump(vectors, f)


def get_similarity(vector1, vector2):
    return np.dot(vector1, vector2)


def get_closest_documents(vector, k, file_path):
    with open(file_path) as f:
        vectors = json.load(f)
    similarities = []
    for v in vectors:
        similarities.append({"similarity": get_similarity(vector, v['vector']), "document": v['document']})
    similarities = pandas.DataFrame(similarities)
    similarities = similarities.sort_values(by='similarity', ascending=False)
    return similarities.head(k).to_dict(orient='records')
