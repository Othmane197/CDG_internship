"""
=============================================================================
CHANTIER 3 — VEILLE SECTORIELLE AUTOMATISÉE (yfinance)
=============================================================================
Auteur    : Stagiaire CDG Invest
Module    : market_watch/veille_sectorielle.py

Description :
    Extrait automatiquement les données de marché pour des entreprises
    cotées servant de "proxys" sectoriels pour vos cibles PE.

    Proxys choisis (modifiables) :
        • Secteur Infrastructure/Utilities (Maroc + international)
        • Secteur Agroalimentaire
        • Secteur Financier / Holding

    Données extraites :
        ✅ Cours de bourse (OHLCV historique)
        ✅ Volumes de trading
        ✅ Indicateurs techniques (MA20, MA50, RSI, Bollinger)
        ✅ Métriques fondamentales (P/E, MarketCap, dividendes)
        ✅ Score de tendance sectorielle

    🔌 Pour adapter vos proxys :
        Modifiez le dictionnaire SECTOR_PROXIES ci-dessous.

=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OUTPUT_DIR   = "reports/outputs"
DATA_RAW_DIR = "data/raw"

# Proxys sectoriels — Remplacez par les tickers les plus pertinents
# pour les secteurs de vos cibles d'investissement
SECTOR_PROXIES = {
    "Infrastructure & Utilities": [
        "ENEL.MI",   # Enel — Infrastructure énergie (proxy international)
        "VIE.PA",    # Veolia — Eau/déchets (très pertinent Maroc/Afrique)
        "ADP.PA",    # Aéroports de Paris — Infrastructure concédée
    ],
    "Agroalimentaire & FMCG": [
        "DANOY",     # Danone — Agroalimentaire (présent au Maroc)
        "NESN.SW",   # Nestlé — Leader mondial
        "UNILV.AS",  # Unilever — FMCG Afrique/MENA
    ],
    "Finance & Holding": [
        "BNP.PA",    # BNP Paribas — Bancaire (partenaire CDG)
        "EDF.PA",    # EDF — Holding énergie publique (analogie CDG)
        "MCO",       # Moody's — Agence de notation (veille crédit)
    ],
}

# Horizon temporel par défaut
DEFAULT_PERIOD = "6mo"   # Options: 1mo, 3mo, 6mo, 1y, 2y, 5y
DEFAULT_INTERVAL = "1d"  # Options: 1d, 1wk, 1mo

plt.style.use("dark_background")
PALETTE = {
    "primary": "#38bdf8",
    "success": "#22c55e",
    "danger" : "#ef4444",
    "warning": "#f59e0b",
    "bg"     : "#0f172a",
    "card"   : "#1e293b",
    "text"   : "#e2e8f0",
    "muted"  : "#64748b",
}


# ---------------------------------------------------------------------------
# SECTION 1 : EXTRACTION DES DONNÉES
# ---------------------------------------------------------------------------

def fetch_market_data(
    tickers: list,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> dict:
    """
    Télécharge les données OHLCV historiques pour une liste de tickers.

    Parameters
    ----------
    tickers  : list — Liste de symboles Yahoo Finance
    period   : str  — Période historique (ex: "6mo", "1y")
    interval : str  — Fréquence des données

    Returns
    -------
    dict {ticker: pd.DataFrame} — OHLCV par ticker
    """
    data = {}
    for ticker in tickers:
        try:
            print(f"   📡 Téléchargement : {ticker}...", end=" ")
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                print("❌ (données vides)")
            else:
                data[ticker] = df
                print(f"✅ {len(df)} séances")
        except Exception as e:
            print(f"❌ Erreur : {e}")
        time.sleep(0.3)  # Respecter les rate limits

    return data


def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Extrait les métriques fondamentales clés pour chaque ticker.

    Returns
    -------
    pd.DataFrame avec colonnes : marketCap, trailingPE, dividendYield,
                                  52WeekHigh, 52WeekLow, beta, sector
    """
    records = []
    fields  = [
        "marketCap", "trailingPE", "forwardPE", "dividendYield",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta",
        "sector", "shortName", "currency",
    ]

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            row  = {"ticker": ticker}
            for f in fields:
                row[f] = info.get(f, None)
            records.append(row)
        except Exception:
            records.append({"ticker": ticker})
        time.sleep(0.2)

    df = pd.DataFrame(records).set_index("ticker")
    return df


# ---------------------------------------------------------------------------
# SECTION 2 : INDICATEURS TECHNIQUES
# ---------------------------------------------------------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques courants sur un DataFrame OHLCV.

    Indicateurs :
        MA20, MA50  : Moyennes mobiles (tendance)
        RSI(14)     : Relative Strength Index (momentum)
        BB_upper/lower : Bandes de Bollinger (volatilité)
        Daily_Return : Rendement journalier

    Parameters
    ----------
    df : pd.DataFrame — OHLCV avec colonne 'Close'

    Returns
    -------
    df enrichi avec indicateurs techniques
    """
    df = df.copy()
    close = df["Close"].squeeze()  # S'assurer que c'est une Series

    # Moyennes mobiles
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # Rendement journalier
    df["Daily_Return"] = close.pct_change()

    # Volatilité glissante (20 jours, annualisée)
    df["Volatility_20d"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

    # RSI(14) — Relative Strength Index
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bandes de Bollinger (20 jours, 2 écarts-types)
    bb_ma  = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = bb_ma + 2 * bb_std
    df["BB_Lower"] = bb_ma - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_ma  # Largeur normalisée

    # Signal de tendance simple
    df["Trend_Signal"] = np.where(
        df["MA20"] > df["MA50"], "HAUSSIER 📈",
        np.where(df["MA20"] < df["MA50"], "BAISSIER 📉", "NEUTRE ➡️")
    )

    return df


# ---------------------------------------------------------------------------
# SECTION 3 : SCORE DE TENDANCE SECTORIELLE
# ---------------------------------------------------------------------------

def compute_sector_trend_score(all_data: dict) -> pd.DataFrame:
    """
    Agrège les signaux de tous les proxys d'un secteur en un score de tendance.

    Score = moyenne pondérée des rendements sur 1m, RSI normalisé,
            position relative aux MA.

    Returns
    -------
    pd.DataFrame — Score de tendance par ticker
    """
    records = []
    for ticker, df in all_data.items():
        if len(df) < 20:
            continue

        close = df["Close"].squeeze()
        ret_1m = close.pct_change(21).iloc[-1]  # Rendement 1 mois
        ret_5d = close.pct_change(5).iloc[-1]   # Rendement 1 semaine
        rsi    = df["RSI_14"].iloc[-1] if "RSI_14" in df.columns else 50

        # Score composite (normalisé de -100 à +100)
        score_ret = np.clip(ret_1m * 200, -50, 50)   # Rendement 1m → [-50, 50]
        score_rsi = (rsi - 50) * 1                    # RSI centré 50 → [-50, 50]
        total_score = round(score_ret + score_rsi, 1)

        current_price = close.iloc[-1]
        ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else None
        ma50 = df["MA50"].iloc[-1] if "MA50" in df.columns else None

        records.append({
            "ticker"          : ticker,
            "current_price"   : round(current_price, 2),
            "ret_1w_pct"      : round(ret_5d * 100, 2),
            "ret_1m_pct"      : round(ret_1m * 100, 2),
            "rsi_14"          : round(rsi, 1),
            "vs_ma20_pct"     : round((current_price / ma20 - 1) * 100, 2) if ma20 else None,
            "vs_ma50_pct"     : round((current_price / ma50 - 1) * 100, 2) if ma50 else None,
            "trend_score"     : total_score,
            "signal"          : "FORT HAUSSIER" if total_score > 20 else
                                "HAUSSIER" if total_score > 5 else
                                "NEUTRE" if total_score > -5 else
                                "BAISSIER" if total_score > -20 else "FORT BAISSIER",
        })

    df_score = pd.DataFrame(records)
    if not df_score.empty:
        df_score = df_score.sort_values("trend_score", ascending=False)
    return df_score


# ---------------------------------------------------------------------------
# SECTION 4 : VISUALISATIONS
# ---------------------------------------------------------------------------

def plot_price_dashboard(all_data: dict, save: bool = True):
    """
    Dashboard multi-graphiques : prix + volume + RSI pour chaque ticker.
    Crée un fichier PNG par ticker.
    """
    for ticker, df in all_data.items():
        if len(df) < 20:
            continue

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(PALETTE["bg"])
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

        # ── Subplot 1 : Prix + MAs + Bollinger ─────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(PALETTE["card"])

        ax1.plot(close.index, close, color=PALETTE["primary"],
                 linewidth=1.5, label="Cours")

        if "MA20" in df:
            ax1.plot(df.index, df["MA20"], color=PALETTE["success"],
                     linewidth=1.2, linestyle="--", label="MA20", alpha=0.9)
        if "MA50" in df:
            ax1.plot(df.index, df["MA50"], color=PALETTE["warning"],
                     linewidth=1.2, linestyle="--", label="MA50", alpha=0.9)
        if "BB_Upper" in df:
            ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                             alpha=0.08, color=PALETTE["primary"])
            ax1.plot(df.index, df["BB_Upper"], color=PALETTE["muted"],
                     linewidth=0.8, linestyle=":", alpha=0.6)
            ax1.plot(df.index, df["BB_Lower"], color=PALETTE["muted"],
                     linewidth=0.8, linestyle=":", alpha=0.6)

        ax1.set_title(f"{ticker} — Analyse Technique",
                      color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax1.set_ylabel("Prix", color=PALETTE["text"])
        ax1.tick_params(colors=PALETTE["text"], labelbottom=False)
        ax1.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
                   fontsize=8, loc="upper left")
        for sp in ax1.spines.values():
            sp.set_edgecolor(PALETTE["card"])

        # ── Subplot 2 : Volume ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor(PALETTE["card"])
        vol_colors = [PALETTE["success"] if r >= 0 else PALETTE["danger"]
                      for r in df["Daily_Return"].fillna(0)]
        ax2.bar(df.index, volume, color=vol_colors, alpha=0.7, width=0.8)
        ax2.set_ylabel("Volume", color=PALETTE["text"], fontsize=9)
        ax2.tick_params(colors=PALETTE["text"], labelbottom=False)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:.0f}")
        )
        for sp in ax2.spines.values():
            sp.set_edgecolor(PALETTE["card"])

        # ── Subplot 3 : RSI ─────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor(PALETTE["card"])

        if "RSI_14" in df:
            rsi = df["RSI_14"]
            ax3.plot(df.index, rsi, color=PALETTE["primary"],
                     linewidth=1.3, label="RSI(14)")
            ax3.axhline(70, color=PALETTE["danger"], linestyle="--",
                        linewidth=1, alpha=0.7, label="Surachat (70)")
            ax3.axhline(30, color=PALETTE["success"], linestyle="--",
                        linewidth=1, alpha=0.7, label="Survente (30)")
            ax3.fill_between(df.index, 30, 70, alpha=0.04,
                             color=PALETTE["primary"])
            ax3.set_ylim(0, 100)
            ax3.set_ylabel("RSI", color=PALETTE["text"], fontsize=9)
            ax3.tick_params(colors=PALETTE["text"])
            ax3.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
                       fontsize=7, loc="upper left")
            for sp in ax3.spines.values():
                sp.set_edgecolor(PALETTE["card"])

        plt.tight_layout()
        if save:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"veille_{ticker.replace('.', '_')}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"   💾 Dashboard {ticker} → {path}")
        else:
            plt.show()
        plt.close()


def plot_sector_heatmap(score_df: pd.DataFrame, save: bool = True):
    """
    Heatmap des performances relatives par ticker / secteur.
    """
    if score_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    cols = ["ret_1w_pct", "ret_1m_pct", "rsi_14", "trend_score"]
    labels = ["Rend. 1S (%)", "Rend. 1M (%)", "RSI(14)", "Score Tendance"]

    data = score_df.set_index("ticker")[cols].copy()
    data.columns = labels

    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    sns.heatmap(
        data, ax=ax, cmap=cmap, center=0,
        annot=True, fmt=".1f", annot_kws={"size": 9},
        linewidths=0.5, linecolor=PALETTE["bg"],
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Heatmap de Tendance — Proxys Sectoriels",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors=PALETTE["text"])
    ax.set_facecolor(PALETTE["card"])

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "veille_sector_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"   💾 Heatmap sectorielle → {path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# SECTION 5 : EXPORT ET SAUVEGARDE
# ---------------------------------------------------------------------------

def export_market_data(all_data: dict, score_df: pd.DataFrame):
    """Sauvegarde les données brutes et le score de tendance en CSV."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    for ticker, df in all_data.items():
        path = os.path.join(DATA_RAW_DIR, f"{ticker.replace('.', '_')}_ohlcv.csv")
        df.to_csv(path)

    if not score_df.empty:
        score_path = os.path.join(DATA_RAW_DIR, "sector_trend_scores.csv")
        score_df.to_csv(score_path, index=False)
        print(f"\n💾 Scores de tendance → {score_path}")


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CDG INVEST — VEILLE SECTORIELLE AUTOMATISÉE (yfinance)")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Aplatir tous les tickers
    all_tickers = [
        ticker
        for sector_tickers in SECTOR_PROXIES.values()
        for ticker in sector_tickers
    ]

    # ── 1. Extraction des données ─────────────────────────────────────────
    print(f"\n📡 Extraction des données marché ({len(all_tickers)} tickers)...")
    all_data = fetch_market_data(all_tickers, period=DEFAULT_PERIOD)

    if not all_data:
        print("❌ Aucune donnée récupérée. Vérifiez votre connexion Internet.")
        return

    # ── 2. Indicateurs techniques ─────────────────────────────────────────
    print("\n⚙️  Calcul des indicateurs techniques...")
    for ticker in all_data:
        all_data[ticker] = compute_technical_indicators(all_data[ticker])

    # ── 3. Métriques fondamentales ────────────────────────────────────────
    print("\n📊 Extraction des données fondamentales...")
    fundamentals = fetch_fundamentals(list(all_data.keys()))
    print(fundamentals[["shortName", "marketCap", "trailingPE",
                          "dividendYield", "beta"]].to_string())

    # ── 4. Score de tendance sectorielle ──────────────────────────────────
    print("\n📈 Calcul des scores de tendance sectorielle...")
    score_df = compute_sector_trend_score(all_data)

    if not score_df.empty:
        print("\n  CLASSEMENT PAR TENDANCE :")
        print(score_df[["ticker", "current_price", "ret_1m_pct",
                          "rsi_14", "trend_score", "signal"]].to_string(index=False))

    # ── 5. Visualisations ─────────────────────────────────────────────────
    print("\n🎨 Génération des dashboards...")
    plot_price_dashboard(all_data)
    plot_sector_heatmap(score_df)

    # ── 6. Export ──────────────────────────────────────────────────────────
    export_market_data(all_data, score_df)

    print("\n✅ Veille sectorielle terminée !")
    print(f"   → Dashboards dans : {OUTPUT_DIR}/")
    print(f"   → Données brutes dans : {DATA_RAW_DIR}/")


if __name__ == "__main__":
    main()
