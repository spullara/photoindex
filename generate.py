import json
import os
import sys

import open_clip
import torch
from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener

register_heif_opener()

model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')


def get_embedding(image: Image):
    # Your function to get the embedding of an image
    with torch.no_grad(), torch.cuda.amp.autocast():
        image = preprocess(image).unsqueeze(0)
        return model.encode_image(image)


def get_image_metadata(image_path: str) -> dict:
    image = Image.open(image_path)
    metadata = {}

    if hasattr(image, '_getexif'):  # Check if the image has EXIF data
        exif_data = image._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                metadata[TAGS.get(tag)] = value

    return metadata


def process_images(directory: str, images_path: str, embeddings) -> None:
    for filename in os.listdir(directory):
        try:
            file_path = os.path.join(directory, filename)

            if os.path.isdir(file_path):
                process_images(file_path, images_path, embeddings)
            elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".heic")):
                print(file_path)
                image = Image.open(file_path)
                metadata = get_image_metadata(file_path)
                embedding = get_embedding(image).cpu().detach().tolist()[0]
                embeddings.append({"path": os.path.relpath(file_path, images_path), "embedding": embedding, "metadata": metadata})
        except Exception as e:
            print(e)


def image_embeddings_to_json(images_path: str, output_path: str) -> None:
    embeddings = []

    process_images(images_path, images_path, embeddings)

    with open(output_path, "w") as output_file:
        json.dump(embeddings, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    images_path = sys.argv[1]
    output_path = "image_embeddings.json"

    image_embeddings_to_json(images_path, output_path)
