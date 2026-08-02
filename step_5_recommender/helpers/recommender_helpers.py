# Helper code for the recommender notebook.

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from IPython.display import Image, Markdown, clear_output, display
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except Exception:
    WIDGETS_AVAILABLE = False


def find_repo_root(start: Path | None = None) -> Path:
    """Takes a start folder and returns the repo root by finding embedding files."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        old_path = candidate / 'text_embeddings' / 'brand_embeddings.npz'
        new_path = candidate / 'step_2_text_embeddings' / 'brand_embeddings.npz'
        if old_path.exists() or new_path.exists():
            return candidate
    raise FileNotFoundError('Could not find the saved embedding file above the current directory.')


ROOT = find_repo_root()
EMBEDDING_DIR = ROOT / 'step_2_text_embeddings'
if not (EMBEDDING_DIR / 'brand_embeddings.npz').exists():
    EMBEDDING_DIR = ROOT / 'text_embeddings'
EMBEDDINGS_PATH = EMBEDDING_DIR / 'brand_embeddings.npz'
METADATA_PATH = EMBEDDING_DIR / 'brand_metadata.csv'
MOODBOARD_ROOT = ROOT / 'moodboards_creation' / 'moodboards'
if not MOODBOARD_ROOT.exists():
    MOODBOARD_ROOT = ROOT / 'step_3_moodboards' / 'moodboards'
RECOMMENDER_DIR = ROOT / 'step_5_recommender'
if not RECOMMENDER_DIR.exists():
    RECOMMENDER_DIR = ROOT / 'recommender'
ROOT



def slugify_brand(name: str) -> str:
    """Takes a brand name and returns a safe file name."""
    text = unicodedata.normalize('NFKD', str(name))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text.removeprefix('and_')


def find_moodboard_path(row: pd.Series) -> Path | None:
    """Takes a brand row and returns its moodboard file path."""
    category = str(row['category'])
    slug = slugify_brand(row['brand_name'])
    direct = MOODBOARD_ROOT / category / f'{slug}.jpg'
    if direct.exists():
        return direct
    category_dir = MOODBOARD_ROOT / category
    if category_dir.exists():
        matches = sorted(category_dir.glob(f'{slug}.*'))
        if matches:
            return matches[0]
    return None


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Takes number rows and returns normalised number rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
embeddings_all = npz['embeddings'].astype('float32')
metadata_all = pd.read_csv(METADATA_PATH)

metadata_all['moodboard_path'] = metadata_all.apply(find_moodboard_path, axis=1)
has_moodboard = metadata_all['moodboard_path'].notna().to_numpy()

catalog = metadata_all.loc[has_moodboard].reset_index(drop=True).copy()
embeddings = l2_normalize(embeddings_all[has_moodboard])
catalog['slug'] = catalog['brand_name'].map(slugify_brand)
catalog['record_id'] = catalog['category'].astype(str) + '::' + catalog['slug']
catalog['moodboard_path'] = catalog['moodboard_path'].map(lambda p: str(Path(p)))

print(f'All embedded brands: {len(metadata_all):,}')
print(f'Brands with generated moodboards: {len(catalog):,}')
print(f'Embedding dimensions: {embeddings.shape[1]:,}')
catalog[['brand_name', 'category', 'aesthetic_keywords', 'moodboard_path']].head()

BATCH_SIZE = 10
MIN_LOVES_FOR_RECOMMENDATIONS = 2
REWARD_VALUES = {
    'No, thanks': 0.0,
    'Maybe': 0.5,
    'Love it': 1.0,
}


def category_catalog(category: str | None = None) -> pd.DataFrame:
    """Takes a category name and returns the moodboard-backed catalogue subset."""
    if category is None or category == 'all':
        return catalog.reset_index(drop=True).copy()
    return catalog[catalog['category'].eq(category)].reset_index(drop=True).copy()


def category_embeddings(category: str | None = None) -> np.ndarray:
    """Takes a category name and returns the aligned embedding subset."""
    if category is None or category == 'all':
        return embeddings
    return embeddings[catalog['category'].eq(category).to_numpy()]


def diverse_probe_indices(X: np.ndarray, count: int = BATCH_SIZE, seed: int | None = 7) -> list[int]:
    """Takes embeddings and returns a far-apart cold-start probe set."""
    if len(X) == 0:
        return []
    rng = np.random.default_rng(seed)
    target = min(count, len(X))
    first = int(rng.integers(len(X)))
    selected = [first]
    selected_set = {first}
    min_distance = 1.0 - (X @ X[first])

    while len(selected) < target:
        candidates = [idx for idx in range(len(X)) if idx not in selected_set]
        if not candidates:
            break
        next_idx = int(max(candidates, key=lambda idx: min_distance[idx]))
        selected.append(next_idx)
        selected_set.add(next_idx)
        min_distance = np.minimum(min_distance, 1.0 - (X @ X[next_idx]))
    return selected


def love_count(feedback: list[tuple[int, float]]) -> int:
    """Takes feedback and returns the number of Love it reactions."""
    return sum(1 for _, reward in feedback if reward >= REWARD_VALUES['Love it'])


def ready_for_recommendations(seen: Iterable[int], feedback: list[tuple[int, float]], batch_size: int = BATCH_SIZE, min_loves: int = MIN_LOVES_FOR_RECOMMENDATIONS) -> bool:
    """Takes session state and returns whether recommendation display is allowed."""
    return len(set(seen)) >= batch_size and love_count(feedback) >= min_loves



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

    cols = ['record_id', 'brand_name', 'category', 'aesthetic_keywords', 'silhouettes', 'materials', 'palette', 'moodboard_path']
    result = catalog.iloc[order][cols].copy()
    result.insert(0, 'rank', range(1, len(result) + 1))
    result.insert(1, 'catalog_index', order)
    result['ucb_score'] = ucb[order].round(4)
    result['exploit'] = exploit[order].round(4)
    result['explore'] = explore[order].round(4)
    return result.reset_index(drop=True)


def show_brand(display_catalog: pd.DataFrame, idx: int, mode: str, progress: str) -> None:
    """Takes a brand index and displays its moodboard plus vibe text."""
    row = display_catalog.loc[idx]
    display(Markdown(f"### {row['brand_name']} ({row['category']})"))
    display(Markdown(f"**{progress}** - {mode}"))
    display(Image(filename=row['moodboard_path'], width=700))
    display(Markdown(f"**Vibe:** {row['aesthetic_keywords']}"))
    display(Markdown(f"**Silhouettes:** {row['silhouettes']}"))
    display(Markdown(f"**Palette:** {row['palette']}"))


def start_recommender(
    category: str = 'clothes',
    batch_size: int = BATCH_SIZE,
    min_loves: int = MIN_LOVES_FOR_RECOMMENDATIONS,
    alpha: float = 0.35,
    ridge: float = 1.0,
    seed: int | None = 7,
    top_k: int = 8,
    max_steps: int | None = None,
):
    """Takes session settings and starts the swipe recommender UI."""
    session_catalog = category_catalog(category)
    session_embeddings = category_embeddings(category)
    if session_catalog.empty:
        raise ValueError(f'No moodboard-backed brands found for category {category!r}.')
    cold_start_queue = diverse_probe_indices(session_embeddings, count=batch_size, seed=seed)
    feedback: list[tuple[int, float]] = []
    seen: set[int] = set()
    current = {'idx': None, 'stopped': False}

    def choose_next() -> tuple[int | None, str]:
        """Takes no input and returns the next brand using diverse probes then LinUCB."""
        while cold_start_queue:
            idx = int(cold_start_queue.pop(0))
            if idx not in seen:
                return idx, 'diverse cold-start probe'
        if len(seen) >= len(session_catalog):
            return None, 'done'
        bandit = fit_bandit(feedback, session_embeddings, alpha=alpha, ridge=ridge)
        recs = rank_brands(bandit, session_embeddings, session_catalog, seen=seen, top_k=1)
        idx = int(recs.loc[0, 'catalog_index'])
        return idx, 'LinUCB pick'

    def show_recommendations(force: bool = False) -> None:
        """Takes current feedback and displays ranked recommendations."""
        if ready_for_recommendations(seen, feedback, batch_size=batch_size, min_loves=min_loves):
            bandit = fit_bandit(feedback, session_embeddings, alpha=alpha, ridge=ridge)
            recs = rank_brands(bandit, session_embeddings, session_catalog, seen=seen, top_k=top_k)
            display(Markdown('### Current recommendations'))
            display(recs[['rank', 'brand_name', 'category', 'ucb_score', 'exploit', 'explore']])
        elif force:
            display(Markdown(
                f'### Not enough feedback for recommendations yet\n\n'
                f'Seen {len(seen)} of {batch_size} probe moodboards and recorded '
                f'{love_count(feedback)} of {min_loves} required `Love it` reactions.'
            ))

    if not WIDGETS_AVAILABLE:
        display(Markdown(
            '**ipywidgets is not installed in this kernel, so this cell is using the text-mode swiper.**\n\n'
            'Type `n` for No, thanks, `m` for Maybe, `l` for Love it, or `q` to quit.'
        ))
        steps = 0
        while max_steps is None or steps < max_steps:
            idx, mode = choose_next()
            current['idx'] = idx
            clear_output(wait=True)
            if idx is None:
                display(Markdown('### No unseen brands left.'))
                break

            show_brand(
                session_catalog,
                idx,
                mode=mode,
                progress=f'{len(feedback)} feedback events; {len(seen)} brands seen; {love_count(feedback)} loves',
            )
            show_recommendations()

            while True:
                choice = input('No, thanks [n], Maybe [m], Love it [l], quit [q]: ').strip().lower()
                if choice in {'n', 'no', 'no thanks', 'no, thanks'}:
                    seen.add(idx)
                    feedback.append((idx, REWARD_VALUES['No, thanks']))
                    break
                if choice in {'m', 'maybe'}:
                    seen.add(idx)
                    feedback.append((idx, REWARD_VALUES['Maybe']))
                    break
                if choice in {'l', 'love', 'love it'}:
                    seen.add(idx)
                    feedback.append((idx, REWARD_VALUES['Love it']))
                    break
                if choice in {'q', 'quit', 'exit'}:
                    return {'category': category, 'feedback': feedback, 'seen': seen}
                print('Please type n, m, l, or q.')
            steps += 1

        clear_output(wait=True)
        if max_steps is not None and steps >= max_steps:
            display(Markdown(f'### Session complete: reached `max_steps={max_steps}`.'))
        show_recommendations(force=True)
        return {'category': category, 'feedback': feedback, 'seen': seen}

    output = widgets.Output()
    no_button = widgets.Button(description='No, thanks', button_style='danger', icon='thumbs-down')
    maybe_button = widgets.Button(description='Maybe', icon='minus')
    love_button = widgets.Button(description='Love it', button_style='success', icon='heart')
    stop_button = widgets.Button(description='Stop', button_style='warning', icon='stop')
    action_buttons = [no_button, maybe_button, love_button, stop_button]
    buttons = widgets.HBox(action_buttons)

    def disable_buttons() -> None:
        """Takes no input and disables the action buttons."""
        for button in action_buttons:
            button.disabled = True

    def finish(reason: str) -> None:
        """Takes a reason and ends the session by showing final recommendations."""
        current['stopped'] = True
        disable_buttons()
        with output:
            clear_output(wait=True)
            display(Markdown(f'### Session complete: {reason}'))
            display(Markdown(f'Swiped {len(seen)} brands and recorded {len(feedback)} feedback events.'))
            show_recommendations(force=True)

    def render_next() -> None:
        """Takes no input and renders the next brand card."""
        if current['stopped']:
            return
        if max_steps is not None and len(seen) >= max_steps:
            finish(f'reached `max_steps={max_steps}`')
            return
        idx, mode = choose_next()
        current['idx'] = idx
        with output:
            clear_output(wait=True)
            if idx is None:
                finish('no unseen brands left')
                return
            show_brand(
                session_catalog,
                idx,
                mode=mode,
                progress=f'{len(feedback)} feedback events; {len(seen)} brands seen; {love_count(feedback)} loves',
            )
            show_recommendations()

    def record(reward: float) -> None:
        """Takes a reward and records the swipe before rendering the next card."""
        idx = current.get('idx')
        if idx is None or current['stopped']:
            return
        seen.add(idx)
        feedback.append((idx, float(reward)))
        render_next()

    no_button.on_click(lambda _: record(REWARD_VALUES['No, thanks']))
    maybe_button.on_click(lambda _: record(REWARD_VALUES['Maybe']))
    love_button.on_click(lambda _: record(REWARD_VALUES['Love it']))
    stop_button.on_click(lambda _: finish('stopped manually'))

    display(buttons, output)
    render_next()
    return {'category': category, 'feedback': feedback, 'seen': seen}
