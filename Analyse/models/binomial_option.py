"""
=============================================================================
CHANTIER 2b — VALORISATION PAR ARBRE BINOMIAL (OPTIONS RÉELLES)
=============================================================================
Auteur    : Stagiaire CDG Invest
Module    : models/binomial_option.py

Description :
    Implémente un modèle d'arbre binomial de Cox-Ross-Rubinstein (CRR)
    pour la valorisation d'options réelles.

    En Private Equity / Infrastructure, les "options réelles" sont des
    flexibilités stratégiques valorisables :
        • Option d'expansion (Call) : droit d'augmenter la taille d'un projet
        • Option d'abandon   (Put)  : droit de vendre/arrêter le projet
        • Option de différer        : droit de retarder l'investissement

    Paramètres clés :
        S0  : Valeur actuelle de l'actif sous-jacent (VAN du projet)
        K   : Prix d'exercice (coût d'investissement additionnel)
        T   : Maturité en années
        r   : Taux sans risque (ex: taux OAT Maroc 10 ans)
        sigma : Volatilité du projet (estimée via proxys sectoriels)
        N   : Nombre de pas de l'arbre

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# SECTION 1 : CONSTRUCTION DE L'ARBRE BINOMIAL CRR
# ---------------------------------------------------------------------------

def build_binomial_tree(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call"
) -> dict:
    """
    Construit l'arbre binomial complet (CRR) et valorise une option.

    Parameters
    ----------
    S0          : float — Valeur actuelle de l'actif (ex: VAN projet en MAD)
    K           : float — Strike / Prix d'exercice
    T           : float — Maturité (en années)
    r           : float — Taux sans risque annuel (ex: 0.035 pour 3.5%)
    sigma       : float — Volatilité annuelle (ex: 0.25 pour 25%)
    N           : int   — Nombre de périodes (pas de temps)
    option_type : str   — "call" (expansion) ou "put" (abandon)

    Returns
    -------
    dict avec :
        'price'        : Prix de l'option (valeur de la flexibilité)
        'asset_tree'   : Arbre des prix de l'actif
        'option_tree'  : Arbre des valeurs de l'option
        'params'       : Paramètres u, d, p (probabilité risque-neutre)
    """
    # --- Paramètres CRR ---
    dt    = T / N                         # Durée d'un pas
    u     = np.exp(sigma * np.sqrt(dt))   # Facteur de hausse
    d     = 1.0 / u                       # Facteur de baisse (CRR symétrique)
    disc  = np.exp(-r * dt)               # Facteur d'actualisation
    p     = (np.exp(r * dt) - d) / (u - d)  # Probabilité risque-neutre

    # Vérification de la condition d'absence d'arbitrage
    assert 0 < p < 1, f"Probabilité invalide p={p:.4f}. Vérifiez les paramètres."

    # --- Construction de l'arbre de l'actif (matrice N+1 x N+1) ---
    asset_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            # Au nœud (i, j) : i montées, (i-j) baisses
            asset_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # --- Valeur intrinsèque à maturité (nœuds terminaux) ---
    option_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(asset_tree[j, N] - K, 0)
        elif option_type == "put":
            option_tree[j, N] = max(K - asset_tree[j, N], 0)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

    # --- Récurrence backward (de la maturité vers aujourd'hui) ---
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Valeur de continuation (actualisation des nœuds suivants)
            continuation = disc * (p * option_tree[j, i + 1] +
                                   (1 - p) * option_tree[j + 1, i + 1])

            # Pour une option américaine, on peut aussi exercer immédiatement
            if option_type == "call":
                intrinsic = max(asset_tree[j, i] - K, 0)
            else:
                intrinsic = max(K - asset_tree[j, i], 0)

            option_tree[j, i] = max(continuation, intrinsic)

    option_price = option_tree[0, 0]

    return {
        "price"      : round(option_price, 2),
        "asset_tree" : asset_tree,
        "option_tree": option_tree,
        "params"     : {"u": u, "d": d, "p": p, "dt": dt, "disc": disc},
        "inputs"     : {"S0": S0, "K": K, "T": T, "r": r,
                        "sigma": sigma, "N": N, "option_type": option_type},
    }


# ---------------------------------------------------------------------------
# SECTION 2 : VISUALISATION DE L'ARBRE (limité à N ≤ 6 pour lisibilité)
# ---------------------------------------------------------------------------

def plot_binomial_tree(result: dict, max_steps: int = 5, save_path: str = None):
    """
    Visualise l'arbre binomial : valeur de l'actif (haut) et de l'option (bas).

    Parameters
    ----------
    result    : dict — Résultat de build_binomial_tree()
    max_steps : int  — Nombre de pas à afficher (max 5-6 pour lisibilité)
    save_path : str  — Chemin pour sauvegarder la figure (None = affichage)
    """
    N_plot     = min(result["inputs"]["N"], max_steps)
    asset_tree = result["asset_tree"]
    opt_tree   = result["option_tree"]
    option_type = result["inputs"]["option_type"]

    fig, ax = plt.subplots(figsize=(3 * (N_plot + 1), 2 * (N_plot + 1)))
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")

    node_radius = 0.3
    COLOR_ASSET  = "#38bdf8"
    COLOR_OPTION = "#f472b6"
    COLOR_EDGE   = "#475569"

    positions = {}  # (step, node_idx) → (x, y)

    for i in range(N_plot + 1):
        n_nodes = i + 1
        for j in range(n_nodes):
            x = i * 2
            y = (n_nodes - 1) * 0.8 - j * 1.6
            positions[(i, j)] = (x, y)

            # Dessin du nœud (cercle)
            circle = plt.Circle((x, y), node_radius, color="#1e293b",
                                 ec=COLOR_ASSET, linewidth=1.5, zorder=3)
            ax.add_patch(circle)

            # Valeur actif
            ax.text(x, y + 0.12, f"{asset_tree[j, i]:,.0f}",
                    ha="center", va="center", fontsize=7,
                    color=COLOR_ASSET, fontweight="bold", zorder=4)
            # Valeur option
            ax.text(x, y - 0.12, f"[{opt_tree[j, i]:,.0f}]",
                    ha="center", va="center", fontsize=7,
                    color=COLOR_OPTION, zorder=4)

            # Arêtes vers les nœuds suivants
            if i < N_plot:
                for dj, label in [(j, "u"), (j + 1, "d")]:
                    x2, y2 = positions.get((i + 1, dj), (i * 2 + 2, None))
                    if y2 is None:
                        y2 = ((i + 2) - 1) * 0.8 - dj * 1.6
                    ax.plot([x + node_radius, x2 - node_radius],
                            [y, y2], color=COLOR_EDGE, linewidth=1, zorder=1)

    # Légende
    p1 = mpatches.Patch(color=COLOR_ASSET,  label="Valeur Actif")
    p2 = mpatches.Patch(color=COLOR_OPTION, label="[Valeur Option]")
    ax.legend(handles=[p1, p2], loc="upper left",
              facecolor="#1e293b", labelcolor="white", fontsize=9)

    title = (f"Arbre Binomial CRR — Option {option_type.title()} "
             f"(S₀={result['inputs']['S0']:,.0f}, K={result['inputs']['K']:,.0f}, "
             f"σ={result['inputs']['sigma']*100:.0f}%, T={result['inputs']['T']}a)")
    ax.set_title(title, color="white", fontsize=11, pad=10)

    # Ajuster les limites
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"📊 Arbre sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# SECTION 3 : ANALYSE DE SENSIBILITÉ (Greeks simplifiés)
# ---------------------------------------------------------------------------

def sensitivity_analysis(base_params: dict, param_name: str,
                          param_range: np.ndarray) -> pd.DataFrame:
    """
    Calcule le prix de l'option pour une plage de valeurs d'un paramètre.
    Utile pour comprendre la sensibilité de la valeur à la volatilité, etc.

    Parameters
    ----------
    base_params : dict        — Paramètres de base (S0, K, T, r, sigma, N, option_type)
    param_name  : str         — Nom du paramètre à faire varier (ex: "sigma")
    param_range : np.ndarray  — Valeurs à tester

    Returns
    -------
    pd.DataFrame avec colonnes [param_name, "option_price"]
    """
    results = []
    for val in param_range:
        params = base_params.copy()
        params[param_name] = val
        res = build_binomial_tree(**params)
        results.append({param_name: val, "option_price": res["price"]})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE AUTONOME
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("   ARBRE BINOMIAL CRR — Valorisation Option Réelle (CDG Invest)")
    print("=" * 70)

    # --- Cas d'usage : Option d'expansion sur un projet infrastructure ---
    params = {
        "S0"         : 100_000_000,   # VAN actuelle du projet : 100 M MAD
        "K"          : 80_000_000,    # Coût d'expansion : 80 M MAD
        "T"          : 3,             # Horizon : 3 ans
        "r"          : 0.035,         # Taux sans risque OAT Maroc : 3.5%
        "sigma"      : 0.25,          # Volatilité sectorielle : 25%
        "N"          : 5,             # 5 pas de temps
        "option_type": "call",        # Option d'expansion = call
    }

    result = build_binomial_tree(**params)

    print(f"\n📌 Paramètres CRR :")
    print(f"   Facteur hausse u     = {result['params']['u']:.4f}")
    print(f"   Facteur baisse d     = {result['params']['d']:.4f}")
    print(f"   Prob. risque-neutre  = {result['params']['p']:.4f}")
    print(f"\n💰 VALEUR DE L'OPTION RÉELLE D'EXPANSION = "
          f"{result['price']:,.2f} MAD")

    # --- Visualisation de l'arbre ---
    os.makedirs("reports/outputs", exist_ok=True)
    plot_binomial_tree(result, max_steps=5,
                       save_path="reports/outputs/binomial_tree.png")

    # --- Analyse de sensibilité à la volatilité ---
    sigma_range = np.linspace(0.10, 0.50, 20)
    sensitivity = sensitivity_analysis(params, "sigma", sigma_range)
    print("\n📈 SENSIBILITÉ À LA VOLATILITÉ (σ):")
    print(sensitivity.to_string(index=False))

    # Option d'abandon (Put) pour comparaison
    params_put = params.copy()
    params_put["option_type"] = "put"
    result_put = build_binomial_tree(**params_put)
    print(f"\n🛑 VALEUR DE L'OPTION D'ABANDON = {result_put['price']:,.2f} MAD")
