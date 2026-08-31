# Fashion Brand Mood Vibe Discovery

This repo is organised around the notebooks in the numbered step folders. The flow is the clean brand dataset, embeddings, optional moodboards, clustering, and the recommender.

The moodboards and recommender are also hosted on Hugging Face: https://huggingface.co/spaces/Zilorina/brand-moodboard-recommender

## Setup

Python packages:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The cleaned public brand-category CSV files are available on Kaggle:

https://www.kaggle.com/datasets/katjazilonova/fashion-brand-mood-vibe-discovery

The submitted results were generated from a frozen 1,689-row analysis snapshot:

| Category | Records |
| --- | ---: |
| Clothes | 746 |
| Shoes | 316 |
| Bags | 301 |
| Jewellery | 326 |

## Notebook Flow

The numbered notebooks are the main project flow. The notebooks do the following:

1. `step_1_curate_dataset/brand_list_curation_v2.ipynb`

   This creates and cleans the original brand dataset, but it takes a while because it queries many data sources, so for normal use it is easier to download the clean Kaggle dataset.

2. `step_2_text_embeddings/brand_text_embeddings.ipynb`

   This loads the clean Kaggle CSVs and creates the embedding files used by clustering and recommendations.

3. `step_3_moodboards/moodboard_generation.ipynb`

   This makes illustrative brand moodboards. They take a while, need `GEMINI_API_KEY`, and are not uploaded here because they are generated images rather than source data. In this repo, they are only used for illustration.

4. `step_4_clustering/brand_latent_space_clustering.ipynb`

   This runs validation and clustering from the embeddings and produces the relevant figures. Step 4 also contains outputs. `step_4_clustering/pca_latent_axis_exploration.ipynb` makes the PCA/latent-axis analysis.

5. `step_5_recommender/brand_contextual_bandit_recommender.ipynb`

   This runs a small manual feedback check for the text-based recommender from the brand records and embeddings.
