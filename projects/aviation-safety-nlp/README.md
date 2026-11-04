# Aviation Safety Incident Mining with NLP (NASA ASRS)

## Project Summary
Apply natural language processing and unsupervised clustering to aviation incident narratives from the NASA Aviation Safety Reporting System (ASRS) to discover and evaluate recurring safety themes.

## Objectives
- Document the narrative data source, access conditions, and fields used.
- Prepare text consistently while retaining a record of transformations.
- Explore and compare suitable text representations and clustering approaches.
- Evaluate cluster quality and interpretability, then review representative narratives for context.
- Report themes as exploratory findings, not as estimates of incident prevalence or causal conclusions.

## Suggested Workflow
1. Verify current data access, use conditions, and documentation for the selected ASRS data.
2. Define the analysis scope and document any inclusion or exclusion criteria.
3. Clean and transform narrative text with a reproducible, versioned pipeline.
4. Build text representations and compare unsupervised clustering approaches.
5. Evaluate clusters using quantitative diagnostics and qualitative review of representative cases.
6. Summarize themes, limitations, and potential reporting biases in `docs/` and `results/`.

## Folder Guide
- `code/`: text processing, feature extraction, clustering, and evaluation code
- `data/`: data source notes, schema, and permitted local data (not committed by default)
- `docs/`: methodology, references, analysis decisions, and limitations
- `results/`: cluster summaries, visualizations, and reports

## Responsible Use
Follow the data provider's terms and protect sensitive information. Avoid publishing raw narratives unless redistribution is explicitly permitted. Clusters are exploratory and may reflect reporting practices, text-processing choices, or model settings; they should not be treated as definitive safety categories.

## Status
Project structure initialized; dataset, analysis, and results have not yet been added.
