# Corrected Moodboards

This folder keeps patched moodboards separate from the original generated set.

## V2 review workflow

- Candidate files live under `v2/<category>/<slug>__candidate_XX.png` and are never loaded by the app or analysis pipeline.
- Automated checks are written to `v2/qc_report.json`. They screen dimensions, aspect ratio, possible repeated regions, and OCR when an OCR executable is available.
- Human review remains mandatory for brand vibe, category accuracy, product authenticity, duplicate products, and accidental text.
- Only an approved file renamed to `v2/<category>/<slug>.png` or `.jpg` is eligible for the corrected-first resolver in `helpers/source_helpers.py`.
- Hosted and iOS-bundled copies must be updated together after approval. Candidate files must never be copied directly into deployment folders.

Current pilot candidates are listed in `v2/pilot_candidates.json`. They are not deployed.

## APPARIS

- Corrected file: `clothes/apparis.png`
- Original file left unchanged: `../moodboards/clothes/apparis.jpg`
- Hosting bundle copy: `../../recommender/hosting_bundle/data/moodboards_corrected/clothes/apparis.png`
- Reason: the original APPARIS row over-emphasized cozy faux fur, pastel brights, and playful knits. The corrected direction uses structured minimal outerwear, restrained neutrals, cruelty-free/sans-leather material cues, and modern utility silhouettes.
- Dataset wording now used:
  - Aesthetic: structured, minimal, conscious, elevated, modern
  - Silhouettes: trench coats, parkas, bomber jackets, rain jackets, workwear jackets
  - Materials: recycled synthetics, sans-leather finishes, technical twill, faux fur accents
  - Palette: black, brown, neutral, white, muted green
