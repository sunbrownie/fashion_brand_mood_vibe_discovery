# Moodboard asset policy

The current repository tree keeps one canonical corrected moodboard per category and brand:

`moodboards_creation/moodboards_corrected/v2/<category>/<brand_slug>.png`

The iOS and Hugging Face exports use the same canonical board as a 1024 × 1536 JPEG named:

`v2__<category>__<brand_slug>.jpg`

Candidate-numbered images, candidate sidecars, generation queues, and candidate-review logs are not retained in the current tree. Earlier committed versions remain recoverable from Git history; no history rewrite is required for normal review rounds.

Review PDFs are evidence artifacts, not asset sources. A new review round replaces the canonical file only after visual inspection, then regenerates the iOS manifest and deployment JPEG.
