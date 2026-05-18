# CDG Invest — Boîte à Outils Quantitative

> **Auteur :** Stagiaire — CDG Invest  
> **Date de création :** Jour 1 de stage  
> **Objectif :** Architecture complète de pipelines quant — prête à recevoir les vraies données d'entreprise.

---

## 🗂️ Arborescence du Projet

```
CDG/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                  ← Données brutes (CSV, Excel entreprises)
│   ├── synthetic/            ← Données synthétiques générées automatiquement
│   └── processed/            ← Données nettoyées/transformées
│
├── models/
│   ├── altman_scoring.py     ← Chantier 1 : Z''-score Altman (scoring PE)
│   └── binomial_option.py    ← Chantier 2b : Arbre binomial (options réelles)
│
├── pipelines/
│   ├── scoring_pipeline.py   ← Pipeline complète scoring + ranking
│   └── monte_carlo.py        ← Chantier 2a : Simulation Monte Carlo
│
├── market_watch/
│   └── veille_sectorielle.py ← Chantier 3 : Veille via yfinance
│
├── notebooks/
│   ├── 01_scoring_PE.ipynb   ← Exploration interactive scoring
│   ├── 02_monte_carlo.ipynb  ← Exploration stress-testing
│   └── 03_veille_marche.ipynb
│
└── reports/
    └── outputs/              ← Tableaux, graphiques exportés
```

---

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Lancer les pipelines

```bash
# 1. Scoring Private Equity (Z''-score Altman)
python pipelines/scoring_pipeline.py

# 2. Monte Carlo Stress-Testing
python pipelines/monte_carlo.py

# 3. Veille Sectorielle
python market_watch/veille_sectorielle.py
```

---

## 🔌 Comment brancher vos vraies données

Chaque module accepte un `pd.DataFrame` standard.  
Remplacez simplement l'appel à `generate_synthetic_data()` par :

```python
import pandas as pd
df = pd.read_csv("data/raw/votre_fichier.csv")
```

Les colonnes attendues sont documentées dans chaque module.
