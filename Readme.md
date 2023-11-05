# Made by Egor Dmitriev | e.dmitriev@innopolis.university | BS20-RO

## Description

This repo presents my tries in making detoxification model. For this task I chose a basic seq2seq model presented at pytorch.org, more advanced pretrained T5 model from google (google/t5-small-ssm), and Llama2-7b (daryl149/llama-2-7b-chat-hf').

Unfortunately the only model that I have managed to run prepely was basic seq2seq model from torch.org. You can see the result of my efforts in the `notebooks/` and `reports/`

## Structure
```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Jupyter notebooks. Naming convention is a number (for ordering),
│                   and a short delimited description, e.g.
│                   "1.0-initial-data-exporation.ipynb"            
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

