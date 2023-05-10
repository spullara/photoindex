import json

import numpy as np
from autofaiss import build_index

# load the embeddings.json file
with open("image_embeddings.json", "r") as f:
    data = json.load(f)

# create a numpy array from "embedding" values in the json file
embeddings = np.float32([np.float32(embedding["embedding"]) for embedding in data])
index, index_infos = build_index(embeddings, index_path="knn.index", index_infos_path="index_infos.json")