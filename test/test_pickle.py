import pickle

file_path = "data/corpus/vector_database/gte/medical_with_answer/passages_embeddings0.pkl"

with open(file_path, "rb") as fin:
    ids, embeddings = pickle.load(fin)

print(embeddings.shape)

# ids和embeddings的长度相同，ids为list，embeddings为np.ndarray
print(len(ids))
print(len(embeddings))

print(ids[0])
print(ids[1])
print(ids[2])
print(ids[90])
print(embeddings[0].shape)

# print(embeddings[0])
