import torch
import clip
from PIL import Image

# Load the CLIP model and the processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
def load_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    return image

# Encode the image and text to their respective embeddings
def get_clip_similarity(image_path, text_input):
    # Load and process the image
    image = load_image(image_path)

    # Encode text input
    text = clip.tokenize([text_input]).to(device)

    # Get embeddings for both
    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(text)

    # Compute cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding)
    return similarity.item()

# Usage
image_path = "bike.jpg"
text_input = "A person on a bike"

clip_score = get_clip_similarity(image_path, text_input)
print(f"Similarity score between the image and text: {clip_score}")


image_path = "cat.jpg"
text_input = "A person on a bike"

clip_score = get_clip_similarity(image_path, text_input)
print(f"Similarity score between the image and text: {clip_score}")
