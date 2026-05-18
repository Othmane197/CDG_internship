"""
=============================================================================
CHANTIER 2a — STRESS-TESTING PAR SIMULATION DE MONTE CARLO
=============================================================================
Auteur    : Stagiaire CDG Invest
Module    : pipelines/monte_carlo.py

Description :
    Simule 10 000 trajectoires de valeur d'un portefeuille pour modéliser
    le risque (VaR, CVaR, distribution des rendements terminaux).

    Modèle : Mouvement Brownien Géométrique (GBM)
        dS = μ·S·dt + σ·S·dW

    Outputs :
        - Value at Risk (VaR) à 95% et 99%
        - Expected Shortfall / CVaR
        - Distribution des rendements terminaux
        - Fan chart des trajectoires
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OUTPUT_DIR = "reports/outputs"
plt.style.use("dark_background")

PALETTE = {
    "primary"  : "#38bdf8",
    "danger"   : "#ef4444",
    "warning"  : "#f59e0b",
    "success"  : "#22c55e",
    "bg"       : "#0f172a",
    "card"     : "#1e293b",
    "text"     : "#e2e8f0",
    "muted"    : "#64748b",
}


# ---------------------------------------------------------------------------
# SECTION 1 : MOTEUR DE SIMULATION
# ---------------------------------------------------------------------------

def run_monte_carlo(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Exécute n_simulations trajectoires GBM pour un actif/portefeuille.

    Parameters
    ----------
    S0            : Valeur initiale (ex: 100 M MAD)
    mu            : Rendement espéré annuel (ex: 0.08 pour 8%)
    sigma         : Volatilité annuelle (ex: 0.20 pour 20%)
    T             : Horizon en années (ex: 3)
    dt            : Pas de temps (ex: 1/252 pour quotidien, 1/12 pour mensuel)
    n_simulations : Nombre de trajectoires (défaut: 10 000)
    seed          : Graine aléatoire

    Returns
    -------
    np.ndarray de forme (n_steps+1, n_simulations)
        Chaque colonne = une trajectoire de valeur du portefeuille
    """
    rng     = np.random.default_rng(seed)
    n_steps = int(T / dt)

    # Matrice de chocs aléatoires (loi normale)
    Z = rng.standard_normal((n_steps, n_simulations))

    # Facteur de croissance à chaque pas (formule exacte GBM)
    # S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
    drift     = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_returns = drift + diffusion

    # Construction des trajectoires (cumulative product)
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp(log_returns[t - 1])

    return paths


# ---------------------------------------------------------------------------
# SECTION 2 : CALCUL DES MÉTRIQUES DE RISQUE
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    paths: np.ndarray,
    S0: float,
    confidence_levels: list = [0.95, 0.99],
) -> dict:
    """
    Calcule les métriques de risque clés à partir des trajectoires.

    Parameters
    ----------
    paths             : np.ndarray (n_steps+1, n_sims) — Trajectoires
    S0                : float — Valeur initiale
    confidence_levels : list  — Niveaux de confiance pour VaR/CVaR

    Returns
    -------
    dict avec VaR, CVaR, statistiques descriptives
    """
    final_values  = paths[-1]
    final_returns = (final_values - S0) / S0  # Rendements terminaux

    metrics = {
        "S0"              : S0,
        "n_simulations"   : paths.shape[1],
        "mean_final_value": np.mean(final_values),
        "median_final"    : np.median(final_values),
        "std_final"       : np.std(final_values),
        "min_final"       : np.min(final_values),
        "max_final"       : np.max(final_values),
        "mean_return_pct" : np.mean(final_returns) * 100,
        "prob_loss_pct"   : np.mean(final_returns < 0) * 100,
        "VaR"             : {},
        "CVaR"            : {},
    }

    for cl in confidence_levels:
        var_pct   = cl * 100
        var_value = np.percentile(final_returns, 100 - var_pct)
        cvar_value = final_returns[final_returns <= var_value].mean()

        metrics["VaR"][f"{var_pct:.0f}%"]  = var_value * 100
        metrics["CVaR"][f"{var_pct:.0f}%"] = cvar_value * 100

    return metrics


# ---------------------------------------------------------------------------
# SECTION 3 : VISUALISATIONS
# ---------------------------------------------------------------------------

def plot_fan_chart(
    paths: np.ndarray,
    T: float,
    dt: float,
    metrics: dict,
    n_display: int = 200,
    save: bool = True,
):
    """
    Fan chart : superpose n_display trajectoires avec les percentiles.
    """
    n_steps = paths.shape[0]
    time_axis = np.linspace(0, T, n_steps)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    # Trajectoires aléatoires (transparentes)
    idx = np.random.choice(paths.shape[1], size=min(n_display, paths.shape[1]),
                           replace=False)
    for i in idx:
        ax.plot(time_axis, paths[:, i], color=PALETTE["primary"],
                alpha=0.03, linewidth=0.5)

    # Percentiles (fan)
    pcts = [5, 25, 50, 75, 95]
    pct_values = np.percentile(paths, pcts, axis=1)
    labels = ["P5", "P25", "Médiane", "P75", "P95"]
    line_colors = [PALETTE["danger"], PALETTE["warning"], PALETTE["primary"],
                   PALETTE["warning"], PALETTE["success"]]

    for i, (pct_val, label, color) in enumerate(zip(pct_values, labels, line_colors)):
        lw = 2.5 if label == "Médiane" else 1.5
        ax.plot(time_axis, pct_val, color=color, linewidth=lw,
                label=label, zorder=5)

    # Zone de confiance 5%-95%
    ax.fill_between(time_axis, pct_values[0], pct_values[-1],
                    alpha=0.08, color=PALETTE["primary"])

    # Ligne S0
    ax.axhline(metrics["S0"], color=PALETTE["muted"],
               linestyle="--", linewidth=1, alpha=0.7, label="Valeur initiale S₀")

    ax.set_xlabel("Horizon (années)", color=PALETTE["text"], fontsize=11)
    ax.set_ylabel("Valeur du Portefeuille (MAD)", color=PALETTE["text"], fontsize=11)
    ax.set_title(
        f"Simulation Monte Carlo ({metrics['n_simulations']:,} trajectoires) — "
        f"μ={metrics['mean_return_pct']:.1f}%",
        color=PALETTE["text"], fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["card"])

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
              fontsize=9, loc="upper left")

    plt.tight_layout()
    if save:
        _save_fig(fig, "mc_fan_chart.png")
    else:
        plt.show()
    plt.close()


def plot_return_distribution(paths: np.ndarray, S0: float,
                              metrics: dict, save: bool = True):
    """
    Distribution des rendements terminaux avec VaR et CVaR annotés.
    """
    final_returns_pct = (paths[-1] - S0) / S0 * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    # Histogramme
    n_bins = 80
    counts, bin_edges = np.histogram(final_returns_pct, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Coloration des barres (rouge pour pertes, vert pour gains)
    bar_colors = [PALETTE["danger"] if c < 0 else PALETTE["success"]
                  for c in bin_centers]
    ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
           color=bar_colors, alpha=0.7, edgecolor=PALETTE["bg"], linewidth=0.3)

    # Ligne de densité (KDE)
    kde_x = np.linspace(final_returns_pct.min(), final_returns_pct.max(), 300)
    kde   = stats.gaussian_kde(final_returns_pct)
    kde_y = kde(kde_x) * len(final_returns_pct) * (bin_edges[1] - bin_edges[0])
    ax.plot(kde_x, kde_y, color=PALETTE["primary"], linewidth=2, label="KDE")

    # VaR et CVaR
    for cl, (var_key, cvar_key) in [
        (95, ("95%", "95%")),
        (99, ("99%", "99%")),
    ]:
        var_val  = metrics["VaR"][var_key]
        cvar_val = metrics["CVaR"][cvar_key]
        color = PALETTE["warning"] if cl == 95 else PALETTE["danger"]
        alpha = 0.9 if cl == 95 else 1.0

        ax.axvline(var_val, color=color, linestyle="--", linewidth=2,
                   alpha=alpha, label=f"VaR {cl}% = {var_val:.1f}%")
        ax.axvline(cvar_val, color=color, linestyle=":",  linewidth=1.5,
                   alpha=alpha * 0.8, label=f"CVaR {cl}% = {cvar_val:.1f}%")

    ax.axvline(0, color=PALETTE["muted"], linewidth=1, alpha=0.5)

    ax.set_xlabel("Rendement Terminal (%)", color=PALETTE["text"], fontsize=11)
    ax.set_ylabel("Fréquence", color=PALETTE["text"], fontsize=11)
    ax.set_title("Distribution des Rendements Terminaux — Monte Carlo",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["card"])

    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    if save:
        _save_fig(fig, "mc_return_distribution.png")
    else:
        plt.show()
    plt.close()


def _save_fig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"   💾 Figure sauvegardée → {path}")


# ---------------------------------------------------------------------------
# SECTION 4 : RAPPORT TEXTE
# ---------------------------------------------------------------------------

def print_risk_report(metrics: dict, params: dict):
    """Affiche un rapport de risque formaté dans le terminal."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  📊 RAPPORT DE RISQUE — SIMULATION MONTE CARLO")
    print(sep)
    print(f"  Paramètres :")
    print(f"    S₀ (Valeur initiale)  = {params['S0']:>15,.0f} MAD")
    print(f"    μ  (Rendement espéré) = {params['mu']*100:>14.1f} %/an")
    print(f"    σ  (Volatilité)       = {params['sigma']*100:>14.1f} %/an")
    print(f"    T  (Horizon)          = {params['T']:>14.1f} ans")
    print(f"    N  (Simulations)      = {metrics['n_simulations']:>15,}")
    print(sep)
    print(f"  Résultats :")
    print(f"    Valeur finale moyenne = {metrics['mean_final_value']:>15,.0f} MAD")
    print(f"    Valeur finale médiane = {metrics['median_final']:>15,.0f} MAD")
    print(f"    Rendement moyen       = {metrics['mean_return_pct']:>14.2f} %")
    print(f"    Probabilité de perte  = {metrics['prob_loss_pct']:>14.2f} %")
    print(sep)
    print(f"  Métriques de Risque :")
    for cl, var_val in metrics["VaR"].items():
        cvar_val = metrics["CVaR"][cl]
        print(f"    VaR  {cl}  = {var_val:>10.2f}% | "
              f"CVaR {cl}  = {cvar_val:>10.2f}%")
    print(sep)


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CDG INVEST — STRESS-TESTING PAR SIMULATION DE MONTE CARLO")
    print("=" * 70)

    # ── Paramètres (à adapter selon votre portefeuille réel) ──────────────
    params = {
        "S0"           : 100_000_000,  # 100 M MAD — Valeur initiale portefeuille
        "mu"           : 0.10,         # 10%/an — Rendement espéré
        "sigma"        : 0.22,         # 22%/an — Volatilité
        "T"            : 3.0,          # 3 ans — Horizon d'investissement
        "dt"           : 1 / 12,       # Pas mensuel (1/252 pour journalier)
        "n_simulations": 10_000,       # 10 000 trajectoires
        "seed"         : 42,
    }

    print(f"\n▶ Lancement de {params['n_simulations']:,} simulations...")
    paths = run_monte_carlo(**params)
    print(f"   → Matrice trajectoires : {paths.shape[0]} pas × {paths.shape[1]} simulations")

    # Métriques
    metrics = compute_risk_metrics(
        paths, params["S0"], confidence_levels=[0.95, 0.99]
    )

    # Rapport
    print_risk_report(metrics, params)

    # Visualisations
    print("\n🎨 Génération des visualisations...")
    plot_fan_chart(paths, params["T"], params["dt"], metrics)
    plot_return_distribution(paths, params["S0"], metrics)

    print("\n✅ Stress-testing terminé !")
    print(f"   → Résultats dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
