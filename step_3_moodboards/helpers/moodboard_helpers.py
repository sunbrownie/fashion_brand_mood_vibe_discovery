# Helper code for the moodboard notebook.

import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

CWD = Path.cwd().resolve()
ROOT = CWD.parent if CWD.name == 'step_3_moodboards' else CWD

source_path = ROOT / 'helpers' / 'source_helpers.py'
exec(compile(source_path.read_text(encoding='utf-8'), str(source_path), 'exec'), globals())
slugify = slugify_brand_name

allow_kaggle_download = os.environ.get('BRAND_VIBE_DOWNLOAD_KAGGLE', '1').strip().lower()
ensure_final_dataset(
    ROOT,
    allow_download=allow_kaggle_download not in {'0', 'false', 'no'},
)

NOTEBOOK_DIR = ROOT / 'step_3_moodboards'
OUTPUT_DIR = ROOT / 'moodboards_creation' / 'moodboards'
CORRECTED_OUTPUT_DIR = ROOT / 'moodboards_creation' / 'moodboards_corrected' / 'v2'
FINAL_DIR = ROOT / 'final_dataset'
OVERRIDES_PATH = NOTEBOOK_DIR / 'config' / 'brand_moodboard_overrides.json'


def _mask_secret(value: str) -> str:
    """Takes a secret and returns a masked version for printing."""
    value = value.strip()
    if len(value) <= 12:
        return '<too short to mask safely>'
    return f'{value[:6]}...{value[-4:]}'


def load_gemini_api_key() -> tuple[str, Path | None]:
    """Takes no input and returns the Gemini key plus the env file path."""
    env_candidates = [
        NOTEBOOK_DIR / '.env',
        ROOT / 'moodboards_creation' / '.env',
    ]

    loaded_from = None
    for env_path in env_candidates:
        if env_path.exists() and load_dotenv is not None:
            load_dotenv(env_path, override=True)
            loaded_from = env_path
            break

    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        searched = ', '.join(str(p) for p in env_candidates)
        raise RuntimeError(
            'GEMINI_API_KEY is not set. Add it to a local .env file. '
            f'Searched: {searched}'
        )

    os.environ['GEMINI_API_KEY'] = api_key
    return api_key, loaded_from


GEMINI_IMAGE_MODEL = 'gemini-3.1-flash-image'
GEMINI_API_KEY = ''
GEMINI_ENV_PATH = None
client = None

print(f'Output root: {OUTPUT_DIR.resolve()}')


def get_gemini_client():
    """Return a Gemini client, loading the API key only when image generation starts."""
    global GEMINI_API_KEY, GEMINI_ENV_PATH, client
    if client is None:
        if genai is None:
            raise RuntimeError(
                'google-genai is not installed. Install requirements.txt before generating moodboards.'
            )
        GEMINI_API_KEY, GEMINI_ENV_PATH = load_gemini_api_key()
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f'Gemini client ready. Model: {GEMINI_IMAGE_MODEL}')
        print(f'Gemini key: {_mask_secret(GEMINI_API_KEY)}')
        print(f'Loaded env from: {GEMINI_ENV_PATH if GEMINI_ENV_PATH else "existing environment"}')
    return client


AESTHETIC_COLS = ['aesthetic_keywords', 'silhouettes', 'materials', 'palette']

brands_df = load_final_dataset(ROOT, drop_empty_vibe_rows=False)
for col in AESTHETIC_COLS:
    if col not in brands_df.columns:
        brands_df[col] = ''
brands_df[AESTHETIC_COLS] = brands_df[AESTHETIC_COLS].fillna('')
if 'moodboard' not in brands_df.columns:
    brands_df['moodboard'] = ''
brands_df['moodboard'] = brands_df['moodboard'].fillna('')
brands_df = brands_df[['brand_name', 'official_website', 'category'] + AESTHETIC_COLS + ['moodboard']]
brands_df = brands_df[brands_df['brand_name'].str.strip() != ''].reset_index(drop=True)

print(f'Total brands loaded: {len(brands_df)}')
brands_df.groupby('category').size().rename('brands')

_CATEGORY_LABEL = {
    'clothes':   'fashion clothing',
    'shoes':     'footwear',
    'bags':      'handbag and accessories',
    'jewellery': 'fine jewellery',
}

_ITEM_LABEL = {
    'clothes':   'clothing pieces',
    'shoes':     'shoes and footwear',
    'bags':      'bags and handbags',
    'jewellery': 'jewellery pieces',
}

_SIGNATURE_FOCUS = {
    'clothes': (
        'hero garments on model or mannequin, folded and hanging pieces, '
        'necklines, cuffs, hems, buttons, knit structure, print scale, styling layers'
    ),
    'shoes': (
        'shoe shapes from multiple angles, soles, stitching, hardware, material grain, '
        'how the footwear is styled on body'
    ),
    'bags': (
        'bag silhouettes, handles, closures, hardware, interior glimpses, leather or textile grain, '
        'scale on body'
    ),
    'jewellery': (
        'jewellery silhouettes, clasp and setting details, metal finish, stones or beads, '
        'scale on skin and layered styling'
    ),
}

_BRAND_SIGNATURE_NOTES = {
    'Apparis': 'structured minimal cruelty-free outerwear: trench coats, parkas, bombers, rain jackets, workwear jackets; black, brown, neutral, white, muted olive palette; avoid candy brights, pastel plush, cozy knit mood, and novelty faux-fur emphasis.',
}

_QUALITY_SUFFIX = (
    'Approachable magazine collage style, high-resolution but not over-polished; it can feel slightly artsy, playful, or lightly cartoonish while keeping products readable. '
    'Natural professional lighting, crisp product detail, tactile textures, rich but believable colour. '
    'No accidental text, captions, labels, watermarks, or unrelated logos. Vertical portrait format.'
)

_AVOID_GENERIC = (
    'Avoid generic minimalist luxury, anonymous beige moodboards, random street-style filler, '
    'items that contradict the brand cues, repeated views of the same product, repeated colourways of one product, '
    'mirrored/rotated duplicates, sterile product-grid layouts, and literal palette swatch strips.'
)

_MODEL_AND_CONTEXT = (
    'Use model casting that matches the brand range: female-only brands should show only female models; '
    'male-only brands should show only male models; brands that read as unisex or both menswear and womenswear '
    'should show both male and female models, not only one gender. '
    'The background can use the general mood colour of the brand. Context is optional. '
    'Use an interior, plant, prop, street detail, landscape, or other space only when it clearly supports the brand atmosphere.'
)


def load_moodboard_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """Load global and brand-specific art direction used by correction runs."""
    if not path.exists():
        return {'global': {}, 'brands': {}}
    return json.loads(path.read_text(encoding='utf-8'))


MOODBOARD_OVERRIDES = load_moodboard_overrides()


def _override_for(name: str, category: str) -> dict:
    """Return a correction override for one category/brand pair."""
    key = f'{category}/{slugify(name)}'
    return MOODBOARD_OVERRIDES.get('brands', {}).get(key, {})


def _correction_direction(name: str, category: str, reviewer_note: str = '') -> str:
    """Turn global policy, an Ellina note, and any brand override into prompt text."""
    global_rules = MOODBOARD_OVERRIDES.get('global', {})
    override = _override_for(name, category)
    parts = []
    if reviewer_note:
        parts.append(f'Reviewer correction brief: {reviewer_note}')
    for label, key in (
        ('Must include', 'must_include'),
        ('Must avoid', 'must_avoid'),
    ):
        values = override.get(key, [])
        if values:
            parts.append(f'{label}: {"; ".join(values)}.')
    for label, key in (
        ('Composition', 'layout'),
        ('Context', 'context'),
        ('Casting', 'casting'),
        ('Duplicate policy', 'duplicate_policy'),
        ('Text policy', 'text_policy'),
        ('Brand-mark policy', 'logo_policy'),
    ):
        value = override.get(key) or global_rules.get(key)
        if value:
            parts.append(f'{label}: {value}')
    return ' '.join(parts)


def _product_diversity_instruction(cat: str, silhouettes: str) -> str:
    """Takes a category and item list and returns prompt text for varied products."""
    pieces = [piece.strip() for piece in silhouettes.split(',') if piece.strip()]
    if pieces:
        examples = '; '.join(pieces[:12])
        return (
            f'Show a varied product mix selected from these piece types: {examples}. '
            'Use one hero tile per product type, with every main product visibly different in shape and category. '
            'Do not repeat the same item, do not show the same product in another colour, and do not use mirrored, rotated, cropped, or alternate-angle duplicates as separate tiles. '
            'If a detail close-up is included, it must support a different hero product and must not become another copy of the same garment/accessory. '
        )
    return (
        f'Show several different {_ITEM_LABEL[cat]} from the category. '
        'Use one hero tile per product type, with every main product visibly different in shape and category. '
        'Do not repeat the same item, do not show the same product in another colour, and do not use mirrored, rotated, cropped, or alternate-angle duplicates as separate tiles. '
    )


def _clean_field(value: object) -> str:
    """Takes a cell value and returns clean text by handling missing values."""
    return str(value).strip() if pd.notna(value) else ''


def _brand_signature_note(name: str) -> str:
    """Takes a brand name and returns any saved brand-specific prompt note."""
    return _BRAND_SIGNATURE_NOTES.get(name, '')


def moodboard_prompt(row: pd.Series, reviewer_note: str = '') -> str:
    """Takes a brand row and returns an image prompt by combining its vibe fields."""
    name = _clean_field(row['brand_name'])
    cat = _clean_field(row['category'])
    kw = _clean_field(row.get('aesthetic_keywords', ''))
    sil = _clean_field(row.get('silhouettes', ''))
    mat = _clean_field(row.get('materials', ''))
    pal = _clean_field(row.get('palette', ''))
    signature_note = _brand_signature_note(name)
    correction_direction = _correction_direction(name, cat, reviewer_note)
    has_data = bool(kw or sil or mat or pal or signature_note)

    base = (
        f'Create a vertical magazine moodboard collage for {name}, '
        f'a {_CATEGORY_LABEL[cat]} brand. Make it feel unmistakably like this brand, '
        f'not a generic fashion collage. '
    )

    if has_data:
        evidence_parts = []
        if kw:
            evidence_parts.append(f'aesthetic keywords: {kw}')
        if sil:
            evidence_parts.append(f'key silhouettes and pieces: {sil}')
        if mat:
            evidence_parts.append(f'materials and surface texture: {mat}')
        if pal:
            evidence_parts.append(f'palette: {pal}')
        if signature_note:
            evidence_parts.append(f'brand-specific cues: {signature_note}')
        evidence = '; '.join(evidence_parts)
        product_mix = _product_diversity_instruction(cat, sil)
        return (
            f'{base}'
            f'Use these concrete cues as visual evidence: {evidence}. '
            f'{product_mix}'
            f'Build the board from 6-9 distinct fragments: {_SIGNATURE_FOCUS[cat]}, '
            f'optional context only when it matches the brand, a few close crops of characteristic details, '
            f'and brand-specific cut-paper devices such as angular crops, torn edges, irregular frames, textile echoes, or simple hand-drawn accents. '
            f'{_MODEL_AND_CONTEXT} '
            f'{correction_direction} '
            f'Before finalizing the collage, check that no product appears twice. '
            f'Keep the composition editorial and cohesive, with varied crop sizes and negative space. '
            f'{_AVOID_GENERIC} {_QUALITY_SUFFIX}'
        )

    product_mix = _product_diversity_instruction(cat, '')
    return (
        f'{base}'
        f'Infer recognizable brand codes and show {_ITEM_LABEL[cat]} rather than abstract vibes. '
        f'{product_mix}'
        f'Build the board from 6-9 distinct fragments: {_SIGNATURE_FOCUS[cat]}, '
        f'optional context only when it matches the brand, plus brand-specific cut-paper devices such as angular crops, torn edges, irregular frames, textile echoes, or simple hand-drawn accents. '
        f'{_MODEL_AND_CONTEXT} '
        f'{correction_direction} '
        f'Before finalizing the collage, check that no product appears twice. '
        f'{_AVOID_GENERIC} {_QUALITY_SUFFIX}'
    )




_MIME_TO_EXT = {
    'image/png':  '.png',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
}


class GeminiBillingError(RuntimeError):
    """Takes a Gemini billing problem and marks it as a custom error."""
    pass


def is_depleted_credit_error(exc: Exception) -> bool:
    """Takes an exception and returns True when it looks like depleted credits."""
    text = str(exc).lower()
    return (
        'resource_exhausted' in text
        and (
            'prepayment credits are depleted' in text
            or 'prepay' in text
            or 'no credits' in text
        )
    )


def gemini_moodboard(prompt: str, max_retries: int = 3) -> tuple[bytes, str]:
    """Takes a prompt and returns image bytes by calling Gemini with retries."""
    gemini_client = get_gemini_client()
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=['IMAGE']),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    return part.inline_data.data, part.inline_data.mime_type
            raise RuntimeError('No image part in response')
        except Exception as exc:
            if is_depleted_credit_error(exc):
                raise GeminiBillingError(
                    'Gemini billing/prepay credits are depleted for the API key currently loaded. '
                    f'Notebook key: {_mask_secret(GEMINI_API_KEY)}. '
                    f'Loaded env from: {GEMINI_ENV_PATH if GEMINI_ENV_PATH else "existing environment"}.'
                ) from exc
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f'  retry {attempt + 1}/{max_retries - 1} after {wait}s ({exc})')
            time.sleep(wait)


def save_moodboard(raw_bytes: bytes, mime_type: str,
                   category: str, brand_name: str,
                   root: Path = OUTPUT_DIR) -> Path:
    """Takes image bytes and returns the saved path by writing a moodboard file."""
    ext  = _MIME_TO_EXT.get(mime_type, '.png')
    path = root / category / (slugify(brand_name) + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    return path


def save_corrected_moodboard(
    raw_bytes: bytes,
    mime_type: str,
    category: str,
    brand_name: str,
    *,
    candidate: int,
    prompt: str,
    review_id: str,
    root: Path = CORRECTED_OUTPUT_DIR,
) -> Path:
    """Save a non-destructive correction candidate plus its provenance sidecar."""
    ext = _MIME_TO_EXT.get(mime_type, '.png')
    stem = f'{slugify(brand_name)}__candidate_{candidate:02d}'
    path = root / category / f'{stem}{ext}'
    if path.exists():
        raise FileExistsError(f'Correction candidate already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    sidecar = path.with_suffix(path.suffix + '.json')
    sidecar.write_text(
        json.dumps(
            {
                'review_id': review_id,
                'brand_name': brand_name,
                'category': category,
                'candidate': candidate,
                'model': GEMINI_IMAGE_MODEL,
                'prompt': prompt,
                'sha256': hashlib.sha256(raw_bytes).hexdigest(),
                'status': 'candidate_generated',
            },
            ensure_ascii=False,
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )
    return path


def save_category_csv(cat: str) -> None:
    """Takes a category name and saves its updated brand CSV."""
    cat_df = brands_df[brands_df['category'] == cat].drop(columns=['category'])
    cat_df.to_csv(FINAL_DIR / f'brands_{cat}.csv', index=False)


print('Helpers defined.')
