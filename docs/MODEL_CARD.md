# Model card

## Intended use

Educational and exploratory sentiment analysis and corpus inspection for Urdu-script text.

## Out of scope

Moderation, employment, credit, health, legal, political targeting, or other consequential decisions. The corpus and model do not establish robust performance in these settings.

## Training

Character TF-IDF (2–5 character n-grams) with balanced logistic regression. Exact preprocessing is documented in the README. A fixed seed is used for split generation and estimator configuration.

## Evaluation

The CLI writes accuracy, per-class precision/recall/F1, and a confusion matrix from the held-out test split. This repository has not run a valid corpus evaluation because the upstream dataset's license is not established. Do not interpret demonstration expectations as model performance.

## Topic output

NMF topic terms are exploratory fragments. Topic relevance for an individual input uses exact substring matching and should be reviewed by a person.
