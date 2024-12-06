import itertools
import numpy as np
from PIL import Image
import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import torch

# Initialize your CLIP and CLAP models and processors
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Initialize your CLAP processor and model here as needed (assuming they are available)
clap_model = AutoModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

def normalize_scores(scores):
    """Normalizes a tensor of scores to a 0-1 range."""
    '''
    min_score = scores.min()
    max_score = scores.max()
    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores
    '''

    # Detach the tensor from the computation graph and convert it to NumPy
    scores = scores.detach() if scores.requires_grad else scores

    # Subtract a constant for numerical stability (e.g., 20)
    exp_scores = torch.exp(scores - scores.max())

    # Normalize the scores within PyTorch
    normalized_scores = exp_scores / (exp_scores.sum())
    print(normalized_scores, scores)
    return scores

def softmax(x):
    exps = np.exp(x - np.mean(x))
    return exps / exps.sum(axis=0)


def compute_imsm_melfusion(image_list, audio_list, text_list):
    # Load images
    images = [Image.open(image) for image in image_list]

    # Load audio files
    audios = [librosa.load(audio, sr=None)[0] for audio in audio_list]

    # Iterate over all combinations of image, audio, and text
    imsm_scores = []
    clip_scores = []
    clap_scores = []
    #for image_idx, audio_idx, text_idx in itertools.product(range(len(images)), range(len(audios)), range(len(text_list))):
    for i in range (0,min(len(images), len(text_list)),15):
        input_image = images[i]
        input_audio = audios[0]
        input_text = [text_list[i]]

        # Process text and image inputs with CLIP processor
        clip_inputs = clip_processor(text=input_text, images=input_image, return_tensors="pt", padding=True)
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image  # image-text similarity score
        probs_clip = logits_per_image

        print(logits_per_image.item())
        print(logits_per_image.softmax(dim=1))
        #probs_clip_normalized = normalize_scores(probs_clip)  # Normalize CLIP scores

        # Process text and audio inputs with CLAP processor
        clap_inputs = clap_processor(text=input_text, audios=[input_audio], return_tensors="pt", padding=True, sampling_rate=48000)
        clap_outputs = clap_model(**clap_inputs)
        logits_per_audio = clap_outputs.logits_per_audio  # audio-text similarity score
        probs_clap = logits_per_audio
        #probs_clap_normalized = normalize_scores(probs_clap)  # Normalize CLAP scores

        # Calculate IMSM score (combining normalized CLIP and CLAP probabilities)
        probs_metric = probs_clip @ probs_clap.T
        imsm_score = probs_metric.softmax(dim=-1)

        # Convert tensors to NumPy arrays for readability
        probs_clip_np = probs_clip.detach().numpy()
        probs_clap_np = probs_clap.detach().numpy()
        imsm_score_np = imsm_score.detach().numpy()

        # Store results in a structured format (image index, audio index, text index, scores)
        imsm_scores.append({
            'image_idx': i + 1,
            'audio_idx':  1,
            'text_idx': i + 1,
            'clip_scores_normalized': probs_clip_np,
            'clap_scores_normalized': probs_clap_np,
            'imsm_score': imsm_score_np
        })

        # Print the scores with labels
        clip_scores.append(probs_clip_np[0][0])
        clap_scores.append(probs_clap_np[0][0])
        print(f"Normalized CLIP Score (Image {i + 1}, Text {i + 1}): {probs_clip_np[0][0]:.4f}")
        print(f"Normalized CLAP Score (Audio {1}, Text {i + 1}): {probs_clap_np[0][0]:.4f}")
        print(f"IMSM Score (Image {i + 1}, Audio {1}, Text {i + 1}): {imsm_score_np[0][0]:.4f}")
        print("----------------------------------------------------")
    normalized_clip_scores = softmax(clip_scores)
    print(clip_scores)
    print('Normalized clip scores:')
    print(np.mean(normalized_clip_scores))

    normalized_clap_scores = softmax(clap_scores)
    print(clap_scores)
    print('Normalized clap scores:')
    print(np.mean(normalized_clap_scores))

    return imsm_scores



def compute_aligned_imsm_melfusion(image_list, audio_list, text_list):
    # Load images
    images = [Image.open(image) for image in image_list]

    # Load audio files
    audios = [librosa.load(audio, sr=None)[0] for audio in audio_list]

    # Ensure the lists are of equal length
    #assert len(images) == len(audios) == len(text_list), "Lists must be of equal length."

    # Initialize list to store IMSM scores for aligned pairs
    imsm_scores = []

    # Process each aligned pair (image, audio, text)
    for i in range(len(images)):
        input_image = images[i]
        input_audio = audios[0]
        input_text = [text_list[i]]

        # Process text and image with CLIP
        clip_inputs = clip_processor(text=input_text, images=input_image, return_tensors="pt", padding=True)
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image  # image-text similarity score
        probs_clip = logits_per_image
        probs_clip_normalized = normalize_scores(probs_clip)  # Normalize CLIP scores

        # Process text and audio with CLAP
        clap_inputs = clap_processor(text=input_text, audios=[input_audio], return_tensors="pt", padding=True,
                                     sampling_rate=48000)
        clap_outputs = clap_model(**clap_inputs)
        logits_per_audio = clap_outputs.logits_per_audio  # audio-text similarity score
        probs_clap = logits_per_audio
        probs_clap_normalized = normalize_scores(probs_clap)  # Normalize CLAP scores

        # Calculate IMSM score for aligned pair (normalized CLIP and CLAP probabilities)
        probs_metric = probs_clip_normalized @ probs_clap_normalized.T
        imsm_score = probs_metric.softmax(dim=-1)

        # Convert tensor to a NumPy array for readability
        imsm_score_np = imsm_score.detach().numpy().flatten()  # Flatten to get a scalar score

        # Append the scalar IMSM score for this aligned pair to the list
        imsm_scores.append(imsm_score_np[0])
        print(f"Normalized CLIP Score (Image {i + 1}, Text {i + 1}): {probs_clip_normalized}")
        print(f"Normalized CLAP Score (Audio { 1}, Text {i + 1}): {probs_clap_normalized}")
        print(f"IMSM Score (Image {i + 1}, Audio { 1}, Text {i + 1}): {imsm_score_np}")
        print("----------------------------------------------------")

    # Convert IMSM scores list to a NumPy array
    imsm_scores_vector = np.array(imsm_scores)

    # Apply softmax to the vector of IMSM scores
    softmax_imsm_scores = np.exp(imsm_scores_vector) / np.sum(np.exp(imsm_scores_vector))

    return softmax_imsm_scores

#Example usage
#image_list = ["image1.png", "image2.png", ...]  # list of image file paths
#audio_list = ["audio1.wav", "audio2.wav", ...]  # list of audio file paths
#text_list = ["text1", "text2", ...]  # list of text inputs

#scores = compute_imsm_melfusion(image_list, audio_list, text_list)
