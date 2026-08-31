# Helper code for the recommender notebook.

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CWD = Path.cwd().resolve()
ROOT = CWD.parent if CWD.name == 'step_5_recommender' else CWD


metadata_all, embeddings_all, _ = load_embeddings(ROOT)
catalog = metadata_all.reset_index(drop=True).copy()
embeddings = embeddings_all
catalog_mode = 'text embeddings'

catalog['slug'] = catalog['brand_name'].map(slugify_brand_name)
catalog['record_id'] = catalog['category'].astype(str) + '::' + catalog['slug']

print(f'All embedded brands: {len(metadata_all):,}')
print(f'Recommender catalog: {len(catalog):,} brands ({catalog_mode})')
print(f'Embedding dimensions: {embeddings.shape[1]:,}')
catalog[['brand_name', 'category', 'aesthetic_keywords']].head()

REWARD_VALUES = {
    'No, thanks': 0.0,
    'Maybe': 0.5,
    'Love it': 1.0,
}


def category_catalog(category: str | None = None) -> pd.DataFrame:
    """Takes a category name and returns that part of the catalogue."""
    if category is None or category == 'all':
        return catalog.reset_index(drop=True).copy()
    return catalog[catalog['category'].eq(category)].reset_index(drop=True).copy()


def category_embeddings(category: str | None = None) -> np.ndarray:
    """Takes a category name and returns the aligned embedding subset."""
    if category is None or category == 'all':
        return embeddings
    return embeddings[catalog['category'].eq(category).to_numpy()]


@dataclass
class LinUCB:
    """Takes user feedback and scores brands for recommendations."""
    dim: int
    alpha: float = 0.35
    ridge: float = 1.0

    def __post_init__(self):
        """Takes model settings and creates the starting score state."""
        self.A = self.ridge * np.eye(self.dim, dtype='float64')
        self.b = np.zeros(self.dim, dtype='float64')

    def update(self, x: np.ndarray, reward: float) -> None:
        """Takes one brand and reward and updates the recommender."""
        x = np.asarray(x, dtype='float64')
        self.A += np.outer(x, x)
        self.b += float(reward) * x

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Takes brand data and returns recommendation scores."""
        X = np.asarray(X, dtype='float64')
        theta = np.linalg.solve(self.A, self.b)
        exploit = X @ theta
        A_inv_X_t = np.linalg.solve(self.A, X.T)
        uncertainty = np.sqrt(np.maximum(np.sum(X.T * A_inv_X_t, axis=0), 0.0))
        explore = self.alpha * uncertainty
        return exploit + explore, exploit, explore


def fit_bandit(feedback: list[tuple[int, float]], X: np.ndarray, alpha: float = 0.35, ridge: float = 1.0) -> LinUCB:
    """Takes feedback and returns a fitted bandit."""
    bandit = LinUCB(dim=X.shape[1], alpha=alpha, ridge=ridge)
    for idx, reward in feedback:
        bandit.update(X[idx], reward)
    return bandit


def rank_brands(bandit: LinUCB, X: np.ndarray, catalog: pd.DataFrame, seen: Iterable[int] = (), top_k: int = 12) -> pd.DataFrame:
    """Takes a fitted bandit and returns top brands by ranking unseen scores."""
    ucb, exploit, explore = bandit.score(X)
    score = ucb.copy()
    seen = set(seen)
    if seen:
        score[list(seen)] = -np.inf
    order = np.argsort(-score)[:top_k]

    cols = ['record_id', 'brand_name', 'category', 'aesthetic_keywords', 'silhouettes', 'materials', 'palette']
    result = catalog.iloc[order][cols].copy()
    result.insert(0, 'rank', range(1, len(result) + 1))
    result.insert(1, 'catalog_index', order)
    result['ucb_score'] = ucb[order].round(4)
    result['exploit'] = exploit[order].round(4)
    result['explore'] = explore[order].round(4)
    return result.reset_index(drop=True)
