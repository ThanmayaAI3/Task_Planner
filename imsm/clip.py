from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load an image and text description
image = Image.open("piano.jpg")
text = "white keys and black keys"

# Preprocess the image and text
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

# Calculate the CLIP embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

# Normalize embeddings and calculate cosine similarity
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
cosine_sim = torch.matmul(image_embeds, text_embeds.T)

clip_score = cosine_sim.item()
print(f"CLIP Score: {clip_score}")
