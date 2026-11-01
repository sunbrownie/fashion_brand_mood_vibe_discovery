# Corrected Moodboards

This folder keeps patched moodboards separate from the original generated set.

## V2 review workflow

- The current tree stores one canonical latest board at `v2/<category>/<slug>.png`.
- Automated checks may screen dimensions, aspect ratio, possible repeated regions, and OCR, but human review remains mandatory for brand vibe, category accuracy, product authenticity, duplicate products, and accidental text.
- A reviewed replacement overwrites the canonical V2 file; the prior committed version remains recoverable from Git history.
- Hosted and iOS-bundled copies must be regenerated together from the canonical file after approval.
- Candidate-numbered files, candidate sidecars, queues, and candidate-review logs are not retained in the repository.

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
