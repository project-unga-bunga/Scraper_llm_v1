import gc
import glob
import os
import random
import sys
import time
from dataclasses import asdict

import hdbscan
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from safetensors import safe_open
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_model import LlmScraper
matplotlib.use('TkAgg')
model_name = "Bielik-7B-v0.1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# NOte: LLM Test prompt gen
# scraper = LlmScraper(model_name,device)
# scraper.model.eval()
# prompt = "Hej, czy możesz ze mną pogadać?"
# llm_response = scraper.prompt_gen(prompt)
# print(llm_response)
# del scraper
# gc.collect()
# time.sleep(1)
# NOte: LLM Test prompt gen
# Note: UMAP Prototype
path = "html"

def random_file_to_string(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        raise FileNotFoundError("No files found in the directory.")
    random_file = random.choice(files)
    file_path = os.path.join(directory, random_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

html_str = random_file_to_string(path)
soup = BeautifulSoup(html_str, 'html.parser')

tag_classes = ['a', 'img', 'script', 'div', 'href', 'url']
elements = []
present_tags = []
for tag in tag_classes:
    found_elements = soup.find_all(tag)
    if found_elements:
        present_tags.append(tag)
        for elem in soup.find_all(tag):
            elements.append({"tag": tag, "html_code": str(elem)})

tag_classes = present_tags

html_pd = pd.DataFrame(elements)
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(3,6))
X_vectors = vectorizer.fit_transform(html_pd['html_code'])

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
X_embedded = reducer.fit_transform(X_vectors)
kmeans = KMeans(n_clusters=len(tag_classes), random_state=42)
clusters = kmeans.fit_predict(X_embedded)
html_pd['cluster'] = clusters


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.scatterplot(
    ax=axes[0],
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    hue=clusters,
    palette='Spectral',
    alpha=0.5,
    s=100
)
axes[0].set_title("HTML Code by Clusters")
axes[0].set_xlabel("UMAP d1")
axes[0].set_ylabel("UMAP d2")
axes[0].legend(title='Cluster', bbox_to_anchor=(1, 1))
axes[0].grid(True)

sns.scatterplot(
    ax=axes[1],
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    hue=html_pd['tag'],
    palette="Set2",
    alpha=0.7,
    s=100
)
axes[1].set_title("HTML Code by Tag")
axes[1].set_xlabel("UMAP d1")
axes[1].set_ylabel("UMAP d2")
axes[1].legend(title='HTML Tag', bbox_to_anchor=(1, 1))
axes[1].grid(True)

plt.tight_layout()
plt.show()