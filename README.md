# rag-eval-lab

Experiments on building a synthetic evaluation dataset for a Turkish-language
RAG (retrieval-augmented generation) assistant, and on chunking strategies for
its source documents.


## Approach

The dataset is built following a six-step workflow:

1. **Profile the corpus**: extract entities and relations.
2. **Define the target mix**: question type, question word, hop count, user, phrasing.
3. **Generate single-hop questions**: category-controlled generation.
4. **Generate multi-hop questions**: graph-based generation.
5. **Add hard cases**: unanswerable, noisy and false-premise questions.
6. **Filter and report**: validate correctness, multi-hop validity and diversity; report per slice.

Tools and methods:
- [RAGAS](https://docs.ragas.io) for knowledge-graph-based test set generation
- A re-implementation of the **DataMorgana** method (Filice et al., 2025) for
  category-controlled question generation
- [Weights & Biases](https://wandb.ai) for experiment tracking

## Repository structure

```
.
├── sandbox/          # learning experiments; may be messy, not meant to transfer
├── eval/             # evaluation-set pipeline, built from what sandbox proves
├── LEARNING_LOG.md   # dated log of what was tried, what happened, what was learned
├── requirements.txt  # pinned dependencies
└── README.md
```

## Corpus

The source documents are **not included** in this repository.

## Setup

To be completed once the first dependencies are added.


## Workflow

- Each experiment is logged to Weights & Biases and summarised in `LEARNING_LOG.md`.

## References

- Filice, S. et al. (2025). *Generating Diverse Q&A Benchmarks for RAG Evaluation
  with DataMorgana.* arXiv:2501.12789.
- RAGAS documentation: Testset Generation. https://docs.ragas.io