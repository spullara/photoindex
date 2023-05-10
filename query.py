import json

import faiss
import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image
from pillow_heif import register_heif_opener


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.has_mps:
    device = torch.device("mps")

@st.cache_resource
def load_model():
    register_heif_opener()
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K', device=device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    index = faiss.read_index("knn.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    # load the embeddings.json file
    with open("image_embeddings.json", "r") as f:
        data = json.load(f)
    return model, tokenizer, index, data


def get_embedding(text):
    with torch.no_grad(), torch.cuda.amp.autocast():
        return model.encode_text(tokenizer(text).to(device))


def search(query):
    query_embedding = get_embedding(query).cpu().detach().tolist()[0]
    results = index.search(np.array([query_embedding]), k=10)
    return results[1][0]


def search_and_show_images(query: str):
    image_paths = [s for s in search(query) if s != -1]
    image_paths = [data[i]["path"] for i in image_paths]

    for path in image_paths:
        image = Image.open(path)
        st.image(image, caption=path, use_column_width=True)


model, tokenizer, index, data = load_model()
st.title("Image Search")

query = st.text_input("Enter your search query")
if st.button("Search"):
    search_and_show_images(query)
