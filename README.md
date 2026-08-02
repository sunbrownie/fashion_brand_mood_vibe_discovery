# Fashion Brand Mood Vibe Discovery

## Dataset

The curated fashion brand mood/vibe dataset is available on Kaggle:
https://www.kaggle.com/datasets/katjazilonova/fashion-brand-mood-vibe-discovery

The analysis dataset contains 1,689 brand-category records:

| Category | Records |
| --- | ---: |
| Clothes | 746 |
| Shoes | 316 |
| Bags | 301 |
| Jewellery | 326 |

## Environment

Packages used:

```bash
python -m pip install pandas numpy requests beautifulsoup4 curl_cffi sentence-transformers scikit-learn umap-learn seaborn matplotlib adjustText tqdm pillow python-dotenv google-genai ipywidgets
```

Sentence model cache:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

Notebook order:

1. `step_1_curate_dataset/brand_list_curation_v2.ipynb`
2. `step_2_text_embeddings/brand_text_embeddings.ipynb`
3. `step_3_moodboards/moodboard_generation.ipynb`
4. `step_4_clustering/brand_latent_space_clustering.ipynb`
5. `step_4_clustering/pca_latent_axis_exploration.ipynb`
6. `step_5_recommender/brand_contextual_bandit_recommender.ipynb`
