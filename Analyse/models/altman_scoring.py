# -*- coding: utf-8 -*-
"""
=============================================================================
CHANTIER 1 -- MODELE DE SCORING ALTMAN Z''-SCORE
=============================================================================
Auteur    : Stagiaire CDG Invest
Module    : models/altman_scoring.py

Description :
    Implemente le Z''-score d'Altman (version pour entreprises non cotees
    ET non manufacturieres -- le modele le plus adapte au Private Equity).

    Formule Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Zones de risque :
        Z'' > 2.60             -> Zone "Safe"       (faible risque de faillite)
        1.10 < Z'' <= 2.60     -> Zone Grise        (risque a surveiller)
        Z'' <= 1.10            -> Zone "Distress"   (risque eleve de faillite)

Colonnes attendues (si donnees reelles) :
    - working_capital        : Fonds de Roulement
    - total_assets           : Total Actif
    - retained_earnings      : Reserves et Report a Nouveau
    - ebit                   : Resultat d'Exploitation (EBIT)
    - book_equity            : Capitaux Propres (valeur comptable)
    - total_liabilities      : Total Dettes

Pour brancher vos vraies donnees :
    Remplacez generate_synthetic_data() par :
    df = pd.read_csv("data/raw/votre_fichier.csv")
=============================================================================
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SECTION 1 : GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Génère un dataset fictif de n entreprises avec des ratios financiers
    réalistes pour simuler un portefeuille de Private Equity.

    Parameters
    ----------
    n    : int — Nombre d'entreprises à générer
    seed : int — Graine aléatoire pour la reproductibilité

    Returns
    -------
    pd.DataFrame avec les colonnes financières brutes
    """
    rng = np.random.default_rng(seed)

    # Noms d'entreprises fictives
    sectors = ["Industrie", "Tech", "Agroalimentaire", "Immobilier", "Services"]
    companies = [f"Entreprise_{i+1:03d}" for i in range(n)]
    sector_col = rng.choice(sectors, size=n)

    # --- Total Actif (entre 5M et 500M MAD) ---
    total_assets = rng.uniform(5_000_000, 500_000_000, n)

    # --- Fonds de Roulement (peut être négatif pour les entreprises en difficulté) ---
    # Entre -20% et +40% du Total Actif
    working_capital = total_assets * rng.uniform(-0.20, 0.40, n)

    # --- Réserves et Report à Nouveau (entre -10% et +35% du Total Actif) ---
    retained_earnings = total_assets * rng.uniform(-0.10, 0.35, n)

    # --- EBIT (entre -8% et +25% du Total Actif) ---
    ebit = total_assets * rng.uniform(-0.08, 0.25, n)

    # --- Capitaux Propres (entre 5% et 60% du Total Actif) ---
    book_equity = total_assets * rng.uniform(0.05, 0.60, n)

    # --- Total Dettes = Total Actif - Capitaux Propres (simplifié) ---
    total_liabilities = total_assets - book_equity

    df = pd.DataFrame({
        "company_name"     : companies,
        "sector"           : sector_col,
        "total_assets"     : total_assets,
        "working_capital"  : working_capital,
        "retained_earnings": retained_earnings,
        "ebit"             : ebit,
        "book_equity"      : book_equity,
        "total_liabilities": total_liabilities,
    })

    return df


# ---------------------------------------------------------------------------
# SECTION 2 : CALCUL DES RATIOS X1 → X4
# ---------------------------------------------------------------------------

def compute_altman_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les quatre ratios du modèle Z''-score d'Altman.

    Ratios :
        X1 = Fonds de Roulement / Total Actif
             → Mesure la liquidité à court terme

        X2 = Réserves et Report à Nouveau / Total Actif
             → Mesure la capacité d'autofinancement historique

        X3 = EBIT / Total Actif
             → Mesure la rentabilité économique

        X4 = Capitaux Propres / Total Dettes
             → Mesure le levier financier (solvabilité)

    Parameters
    ----------
    df : pd.DataFrame — Doit contenir les colonnes brutes

    Returns
    -------
    df enrichi avec les colonnes X1, X2, X3, X4
    """
    df = df.copy()

    # Sécurité : éviter la division par zéro
    eps = 1e-9

    df["X1"] = df["working_capital"] / (df["total_assets"] + eps)
    df["X2"] = df["retained_earnings"] / (df["total_assets"] + eps)
    df["X3"] = df["ebit"] / (df["total_assets"] + eps)
    df["X4"] = df["book_equity"] / (df["total_liabilities"] + eps)

    return df


# ---------------------------------------------------------------------------
# SECTION 3 : CALCUL DU Z''-SCORE ET CLASSIFICATION
# ---------------------------------------------------------------------------

ALTMAN_WEIGHTS = {
    "X1": 6.56,
    "X2": 3.26,
    "X3": 6.72,
    "X4": 1.05,
}

SAFE_THRESHOLD     = 2.60
DISTRESS_THRESHOLD = 1.10


def compute_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le Z''-score d'Altman pour chaque entreprise et lui attribue
    une zone de risque.

    Parameters
    ----------
    df : pd.DataFrame — Doit contenir les colonnes X1, X2, X3, X4

    Returns
    -------
    df enrichi avec 'z_score' et 'risk_zone'
    """
    df = df.copy()

    df["z_score"] = (
        ALTMAN_WEIGHTS["X1"] * df["X1"] +
        ALTMAN_WEIGHTS["X2"] * df["X2"] +
        ALTMAN_WEIGHTS["X3"] * df["X3"] +
        ALTMAN_WEIGHTS["X4"] * df["X4"]
    )

    # Classification en zones de risque
    def classify_risk(z: float) -> str:
        if z > SAFE_THRESHOLD:
            return "Safe"
        elif z > DISTRESS_THRESHOLD:
            return "Zone Grise"
        else:
            return "Distress"

    df["risk_zone"] = df["z_score"].apply(classify_risk)

    return df


# ---------------------------------------------------------------------------
# SECTION 4 : RAPPORT DE SCORING FINAL
# ---------------------------------------------------------------------------

def build_scoring_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produit le tableau de classement final des entreprises par niveau de risque.
    Trié du plus risqué au moins risqué (Z'' croissant).

    Parameters
    ----------
    df : pd.DataFrame — Doit contenir 'z_score', 'risk_zone', etc.

    Returns
    -------
    pd.DataFrame — Tableau formaté pour présentation / export
    """
    report = df[[
        "company_name", "sector", "z_score", "risk_zone",
        "X1", "X2", "X3", "X4"
    ]].copy()

    # Tri : les plus risquées en premier (Z'' le plus bas)
    report = report.sort_values("z_score", ascending=True).reset_index(drop=True)
    report.index += 1  # Rang commence à 1

    # Arrondir pour lisibilité
    float_cols = ["z_score", "X1", "X2", "X3", "X4"]
    report[float_cols] = report[float_cols].round(4)

    return report


# ---------------------------------------------------------------------------
# SECTION 5 : STATISTIQUES RÉSUMÉES
# ---------------------------------------------------------------------------

def summarize_risk_distribution(report: pd.DataFrame) -> dict:
    """
    Calcule la répartition du portefeuille par zone de risque.

    Returns
    -------
    dict avec counts et pourcentages par zone
    """
    counts = report["risk_zone"].value_counts()
    total  = len(report)
    summary = {
        zone: {"count": int(cnt), "pct": round(cnt / total * 100, 1)}
        for zone, cnt in counts.items()
    }
    return summary


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE AUTONOME (pour test rapide)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("   ALTMAN Z''-SCORE -- Modele de Scoring PE (CDG Invest)")
    print("=" * 70)

    df_raw = generate_synthetic_data(n=100)
    print(f"\n[OK] Dataset genere : {len(df_raw)} entreprises\n")

    df_ratios = compute_altman_ratios(df_raw)
    df_scored = compute_z_score(df_ratios)
    report    = build_scoring_report(df_scored)

    print("TOP 10 ENTREPRISES LES PLUS A RISQUE :")
    print(report.head(10).to_string())

    print("\nTOP 10 ENTREPRISES LES PLUS SAINES :")
    print(report.tail(10).to_string())

    print("\nREPARTITION PAR ZONE DE RISQUE :")
    dist = summarize_risk_distribution(report)
    for zone, stats in dist.items():
        print(f"  {zone} : {stats['count']} entreprises ({stats['pct']}%)")

    import os
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/scoring_report.csv"
    report.to_csv(output_path)
    print(f"\n[SAVED] Rapport exporte -> {output_path}")
