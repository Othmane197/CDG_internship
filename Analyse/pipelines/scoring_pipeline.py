# -*- coding: utf-8 -*-
"""
=============================================================================
CHANTIER 1 -- PIPELINE DE SCORING PRIVATE EQUITY (ORCHESTRATEUR)
=============================================================================
Auteur    : Stagiaire CDG Invest
Module    : pipelines/scoring_pipeline.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.altman_scoring import (
    generate_synthetic_data,
    compute_altman_ratios,
    compute_z_score,
    build_scoring_report,
    summarize_risk_distribution,
    SAFE_THRESHOLD,
    DISTRESS_THRESHOLD,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

USE_SYNTHETIC_DATA = True           # Passer à False pour les vraies données
DATA_SOURCE_PATH   = "data/raw/portfolio_companies.csv"
N_SYNTHETIC        = 100
OUTPUT_DIR         = "reports/outputs"

plt.style.use("dark_background")
PALETTE = {
    "safe"    : "#22c55e",
    "grey"    : "#f59e0b",
    "distress": "#ef4444",
    "accent"  : "#38bdf8",
    "bg"      : "#0f172a",
    "card"    : "#1e293b",
    "text"    : "#e2e8f0",
}


# ---------------------------------------------------------------------------
# ÉTAPES DE LA PIPELINE
# ---------------------------------------------------------------------------

def ingest_data() -> pd.DataFrame:
    """
    Charge les données selon la configuration.

    🔌 Pour données réelles :
        Mettez USE_SYNTHETIC_DATA = False et renseignez DATA_SOURCE_PATH.
        Colonnes attendues :
            company_name, sector, total_assets, working_capital,
            retained_earnings, ebit, book_equity, total_liabilities
    """
    if USE_SYNTHETIC_DATA:
        print("[INFO] Mode SYNTHETIQUE -- generation de donnees fictives")
        df = generate_synthetic_data(n=N_SYNTHETIC)
    else:
        print(f"[INFO] Chargement depuis : {DATA_SOURCE_PATH}")
        if not os.path.exists(DATA_SOURCE_PATH):
            raise FileNotFoundError(
                f"Fichier non trouve : {DATA_SOURCE_PATH}\n"
                "Verifiez le chemin ou activez USE_SYNTHETIC_DATA = True"
            )
        df = pd.read_csv(DATA_SOURCE_PATH)
        required = ["company_name", "sector", "total_assets", "working_capital",
                    "retained_earnings", "ebit", "book_equity", "total_liabilities"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    print(f"   -> {len(df)} entreprises chargees\n")
    return df


def run_scoring_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Enchaîne les étapes de calcul : ratios → Z'' → rapport."""
    print("[1/3] Calcul des ratios Altman (X1->X4)...")
    df = compute_altman_ratios(df)
    print("[2/3] Calcul des Z''-scores...")
    df = compute_z_score(df)
    print("[3/3] Classement des entreprises...")
    return build_scoring_report(df)


def _save_fig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"   [SAVED] -> {path}")


def plot_risk_distribution(report: pd.DataFrame):
    """Histogramme des Z'' + camembert des zones."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax1, ax2 = axes
    scores = report["z_score"]

    # ── Histogramme ────────────────────────────────────────────────────
    ax1.set_facecolor(PALETTE["card"])
    ax1.axvspan(scores.min() - 0.5, DISTRESS_THRESHOLD,
                alpha=0.15, color=PALETTE["distress"], label="Distress (<1.10)")
    ax1.axvspan(DISTRESS_THRESHOLD, SAFE_THRESHOLD,
                alpha=0.15, color=PALETTE["grey"],    label="Zone Grise")
    ax1.axvspan(SAFE_THRESHOLD, scores.max() + 0.5,
                alpha=0.15, color=PALETTE["safe"],    label="Safe (>2.60)")
    ax1.axvline(DISTRESS_THRESHOLD, color=PALETTE["distress"],
                linestyle="--", linewidth=1.5)
    ax1.axvline(SAFE_THRESHOLD, color=PALETTE["safe"],
                linestyle="--", linewidth=1.5)
    ax1.hist(scores, bins=25, color=PALETTE["accent"],
             edgecolor=PALETTE["bg"], alpha=0.85, linewidth=0.5)
    ax1.set_xlabel("Z''-Score", color=PALETTE["text"], fontsize=11)
    ax1.set_ylabel("Nombre d'entreprises", color=PALETTE["text"], fontsize=11)
    ax1.set_title("Distribution des Z''-Scores", color=PALETTE["text"],
                  fontsize=13, fontweight="bold")
    ax1.tick_params(colors=PALETTE["text"])
    ax1.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"], fontsize=9)
    for sp in ax1.spines.values():
        sp.set_edgecolor(PALETTE["card"])

    # ── Camembert ──────────────────────────────────────────────────────
    ax2.set_facecolor(PALETTE["card"])
    dist = report["risk_zone"].value_counts()
    zone_colors = {
        "Safe"       : PALETTE["safe"],
        "Zone Grise" : PALETTE["grey"],
        "Distress"   : PALETTE["distress"],
    }
    colors = [zone_colors.get(z, PALETTE["accent"]) for z in dist.index]
    wedges, texts, autotexts = ax2.pie(
        dist.values, labels=dist.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
        textprops={"color": PALETTE["text"]},
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontweight("bold")
    ax2.set_title("Répartition par Zone de Risque",
                  color=PALETTE["text"], fontsize=13, fontweight="bold")

    plt.suptitle("CDG Invest — Analyse Altman Z'' (Private Equity)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "scoring_distribution.png")
    plt.close()


def plot_ranking_chart(report: pd.DataFrame, n: int = 20):
    """Barplot horizontal du classement des n entreprises."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    top = report.head(n)

    def zone_color(z):
        if z > SAFE_THRESHOLD:
            return PALETTE["safe"]
        elif z > DISTRESS_THRESHOLD:
            return PALETTE["grey"]
        return PALETTE["distress"]

    colors = [zone_color(z) for z in top["z_score"]]
    bars = ax.barh(top["company_name"], top["z_score"],
                   color=colors, edgecolor=PALETTE["bg"], height=0.7)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{row['z_score']:.2f}", va="center", ha="left",
                color=PALETTE["text"], fontsize=8)

    ax.axvline(DISTRESS_THRESHOLD, color=PALETTE["distress"], linestyle="--",
               linewidth=1.5, label=f"Seuil Distress ({DISTRESS_THRESHOLD})")
    ax.axvline(SAFE_THRESHOLD, color=PALETTE["safe"], linestyle="--",
               linewidth=1.5, label=f"Seuil Safe ({SAFE_THRESHOLD})")

    ax.set_xlabel("Z''-Score", color=PALETTE["text"], fontsize=11)
    ax.set_title(f"Top {n} Entreprises — du Plus Risqué au Moins Risqué",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    ax.tick_params(colors=PALETTE["text"])
    ax.invert_yaxis()
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"], fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["card"])

    plt.tight_layout()
    _save_fig(fig, "scoring_ranking.png")
    plt.close()


def plot_ratios_heatmap(report: pd.DataFrame, n: int = 30):
    """Heatmap des ratios X1-X4 pour les n premières entreprises."""
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor(PALETTE["bg"])

    data = report.head(n)[["X1", "X2", "X3", "X4"]].copy()
    data.index = report.head(n)["company_name"].values

    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    sns.heatmap(data, ax=ax, cmap=cmap, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor=PALETTE["bg"],
                cbar_kws={"label": "Valeur du ratio"})
    ax.set_title(f"Heatmap des Ratios Altman — Top {n}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors=PALETTE["text"])
    ax.set_facecolor(PALETTE["card"])

    plt.tight_layout()
    _save_fig(fig, "scoring_heatmap.png")
    plt.close()


def export_results(report: pd.DataFrame):
    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/scoring_report.csv"
    report.to_csv(path)
    print(f"\n[SAVED] Rapport CSV -> {path}")


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CDG INVEST -- PIPELINE DE SCORING PE (Z''-Score Altman)")
    print("=" * 70 + "\n")

    df      = ingest_data()
    report  = run_scoring_pipeline(df)

    # Résumé console
    print("\n" + "-" * 60)
    print("REPARTITION PAR ZONE DE RISQUE :")
    for zone, stats in summarize_risk_distribution(report).items():
        print(f"  {zone:<22} {stats['count']:>3} entreprises ({stats['pct']:.1f}%)")

    print("\nBOTTOM 10 -- Plus a risque :")
    print(report.head(10)[
        ["company_name", "sector", "z_score", "risk_zone"]
    ].to_string())

    print("\nTOP 10 -- Plus saines :")
    print(report.tail(10)[
        ["company_name", "sector", "z_score", "risk_zone"]
    ].to_string())

    # Visualisations
    print("\n[VIZ] Generation des visualisations...")
    plot_risk_distribution(report)
    plot_ranking_chart(report, n=20)
    plot_ratios_heatmap(report, n=30)

    # Export
    export_results(report)

    print(f"\n[OK] Pipeline terminee ! Resultats -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
