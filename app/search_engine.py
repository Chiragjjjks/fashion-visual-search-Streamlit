# app/search_engine.py
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from ast import literal_eval


# Load model, processor, and data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

products_df = pd.read_csv("C:/Users/chira/Desktop/fashion-visual-search/fashion-visual-search-Streamlit/app/data/valid_products.csv")
embeddings = np.load("C:/Users/chira/Desktop/fashion-visual-search/fashion-visual-search-Streamlit/app/model/fashion_image_embeddings.npy")

# Normalize embeddings for cosine similarity
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS index for fast similarity search
index = faiss.IndexFlatIP(normalized_embeddings.shape[1])  # Inner Product = Cosine similarity with normalized vectors
index.add(normalized_embeddings)

# --------- Get embedding for query image -----------
def get_image_embedding(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # Normalize embedding vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]
    except Exception as e:
        print(f"❌ Error fetching image: {e}")
        return None

# ---------- Search for similar products -------------
def search_similar(image_url, top_k=5, exact_threshold=0.05):
    query_embed = get_image_embedding(image_url)
    if query_embed is None:
        return None

    # Normalize and reshape for FAISS search
    query_embed = query_embed / np.linalg.norm(query_embed)
    query_embed = np.expand_dims(query_embed, axis=0)

    distances, indices = index.search(query_embed, top_k)
    results = products_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    results["exact_match"] = results["distance"] >= (1 - exact_threshold)

    return results[["pdp_images_s3", "category_type", "image_url", "distance", "exact_match"]]

# --------- Display images in matplotlib grid --------------
def display_similar_results(query_image_url, results_df, top_k=5):
    """
    Display the query image and all images of top-k similar results in a grid.
    """
    # Number of rows: one for query, one per result
    n_rows = top_k + 1
    plt.figure(figsize=(20, 3 * n_rows))

    # Display query image first
    try:
        response = requests.get(query_image_url)
        query_img = Image.open(BytesIO(response.content)).convert("RGB")
        ax = plt.subplot(n_rows, 1, 1)
        plt.imshow(query_img)
        plt.axis("off")
        ax.set_title("Query Image", fontsize=12)
    except:
        print("❌ Failed to load query image.")

    # Filter out query image from results
    unique_results = results_df[results_df["image_url"] != query_image_url].head(top_k)

    # Now plot all images for each result
    for i, (_, row) in enumerate(unique_results.iterrows()):
        try:
            # Parse stringified list safely
            img_urls = literal_eval(row["pdp_images_s3"]) if isinstance(row["pdp_images_s3"], str) else []

            for j, img_url in enumerate(img_urls):
                ax = plt.subplot(n_rows, max(len(img_urls), 1), (i + 1) * max(len(img_urls), 1) + j + 1)
                try:
                    response = requests.get(img_url, timeout=10)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    plt.imshow(img)
                    plt.axis("off")
                    if j == 0:
                        ax.set_title(f"{row['category_type']} (Dist: {row['distance']:.2f})", fontsize=9)
                except Exception as e:
                    print(f"❌ Error loading image {j+1} for result {i+1}: {e}")

        except Exception as e:
            print(f"❌ Error parsing pdp_images_s3: {e}")

    plt.tight_layout()
    plt.show()

# --------- Example usage --------------
image_url = "https://gallery.stylumia.com/originals/2020/02/14/b613d7b5dfe86f3e695d931d31fd729fdf44e181f14079d3d8ca9082e8414683_5.jpg"
similar_items = search_similar(image_url, top_k=10)

if similar_items is not None:
    display_similar_results(image_url, similar_items, top_k=5)
else:
    print("Failed to retrieve similar items.")