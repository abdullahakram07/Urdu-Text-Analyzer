# Dataset handling notes

## Upstream reference

- Name: Urdu Sentiment Corpus v1 (tweet corpus)
- Source: https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus
- File: `urdu-sentiment-corpus-v1.tsv`
- Observed schema: header `Tweet` and `Class`; labels `P`, `N`, `O`
- Associated work: Khan & Nizami (2020), and Khan, Emaduddin & Junejo (2017); citations are in the upstream README.
- License: no license file or explicit data license was present in the upstream repository as checked on 2026-10-19. Ask the authors/rightsholder for terms before downloading or using it. No dataset copy is included here.

## Preparation

When authorized, obtain the TSV from upstream and run:

```bash
urdu-analyzer prepare path/to/urdu-sentiment-corpus-v1.tsv --output data/splits --seed 42
```

The loader validates UTF-8 and schema and maps P/N/O to positive/negative/neutral. It retains normalized Unicode text. The split procedure groups identical normalized text to prevent duplicate leakage and rejects duplicates with conflicting labels. It then approximately stratifies groups by label. Since the corpus is small, rare labels may not occur in all three subsets; inspect the reported split sizes and class distributions before interpreting test metrics. Never choose model settings based on test results.

Generated files are ignored by Git. Do not commit third-party rows unless the source terms expressly permit redistribution.
