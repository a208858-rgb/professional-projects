# Professional Projects

A portfolio workspace for documenting project code, data workflows, results, and decisions. Each project has its own area so that its documentation and outputs stay organized and reproducible.

## Projects

### 1. Machine Learning Identification of Gut Microbial Biomarkers for Obesity Prediction
Analyze gut microbiome data to classify weight status and identify bacterial features associated with obesity.

[Open project overview](projects/gut-microbiome-obesity/README.md)

### 2. Aviation Safety Incident Mining with NLP (NASA ASRS)
Apply natural language processing and unsupervised clustering to aviation incident narratives to discover and evaluate safety themes.

[Open project overview](projects/aviation-safety-nlp/README.md)

## Repository Structure

```text
professional-projects/
├── README.md
├── .gitignore
└── projects/
    ├── gut-microbiome-obesity/
    │   ├── README.md
    │   ├── code/
    │   ├── data/
    │   ├── docs/
    │   └─��� results/
    └── aviation-safety-nlp/
        ├── README.md
        ├── code/
        ├── data/
        ├── docs/
        └── results/
```

## Working With Data

- Keep source data out of Git unless its license, privacy requirements, and repository settings allow redistribution.
- Prefer documented download instructions or data access procedures over committing large or sensitive datasets.
- Keep credentials, private records, and personally identifiable information out of this repository.
- Record preprocessing and analysis steps so results can be reproduced.

## Project Documentation
Each project README describes its objective, proposed workflow, and expected outputs. Add methods, assumptions, citations, and limitations in that project's `docs/` folder as the work develops.
