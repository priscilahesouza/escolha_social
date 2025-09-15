import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, entropy
import itertools
import math
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helpers ---

def visibility_curve(m, rho=0.15):
    # Exponential decay alpha_k = exp(-rho*(k-1))
    k = np.arange(1, m+1)
    alpha = np.exp(-rho*(k-1))
    return alpha


def generate_environment(N=200, m=30, G=6, dims=1, seed=None):
    rng = np.random.default_rng(seed)
    # Users ideal points
    if dims == 1:
        mu = rng.uniform(-1, 1, size=(N, 1))
        X = rng.uniform(-1, 1, size=(m, 1))
    else:
        mu = rng.uniform(-1, 1, size=(N, 2))
        X = rng.uniform(-1, 1, size=(m, 2))
    # item groups (sources): imbalanced Zipf-like
    probs = np.array([1/(g+0.5) for g in range(G)])
    probs = probs/probs.sum()
    groups = rng.choice(G, size=m, p=probs)
    # virality heavy-tailed
    z = np.exp(rng.normal(0.0, 1.0, size=m))  # log-normal mean=1
    return mu, X, groups, z


def utilities(mu, X, sigma=0.5):
    # Gaussian kernel utility in [0,1]
    # u_ij = exp(-||x_j - mu_i||^2 / (2*sigma^2))
    dists = cdist(mu, X, metric='euclidean')
    u = np.exp(-(dists**2) / (2*sigma**2))
    return u


def individual_rankings(u):
    # returns per-user ranking indices from best to worst
    return np.argsort(-u, axis=1)


def borda_aggregate(rankings):
    # rankings: N x m indices sorted from best to worst per user
    N, m = rankings.shape
    # Borda points: m-1 for top, ..., 0 for last
    points = np.zeros(m)
    for r in rankings:
        # r is array of item indices ordered
        # assign scores
        scores = np.arange(m-1, -1, -1)
        points[r] += scores
    ordering = np.argsort(-points)
    return ordering, points


def pairwise_matrix(rankings, m):
    # Majority matrix M[a,b] = #users preferring a over b
    N = rankings.shape[0]
    pos = np.zeros((N, m), dtype=int)
    for i in range(N):
        pos[i, rankings[i]] = np.arange(m)
    M = np.zeros((m, m), dtype=int)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            # users where a ranked before b
            M[a, b] = np.sum(pos[:, a] < pos[:, b])
    return M


def copeland_ranking(M):
    m = M.shape[0]
    wins = np.zeros(m)
    for a in range(m):
        for b in range(m):
            if a == b: continue
            if M[a, b] > M[b, a]:
                wins[a] += 1
            elif M[a, b] == M[b, a]:
                wins[a] += 0.5
    ordering = np.argsort(-wins)
    return ordering, wins


def maximin_ranking(M):
    m = M.shape[0]
    minima = np.zeros(m)
    for a in range(m):
        minima[a] = np.min(M[a, :])
    ordering = np.argsort(-minima)
    return ordering, minima


def condorcet_winner(M):
    m = M.shape[0]
    for a in range(m):
        if np.all([(a==b) or (M[a,b] > M[b,a]) for b in range(m)]):
            return a
    return None


def platform_scores(u, z, beta=4.0, gamma=1.0):
    # Engagement per item: mean_i sigmoid(beta*(u_ij-0.5)) times virality^gamma
    # center u around 0.5 to strengthen extremes
    s = 1/(1 + np.exp(-beta*(u - 0.5)))
    e = s.mean(axis=0) * (z ** gamma)
    return e


def rerank_with_diversity(score, groups, alpha, lam=0.0):
    # Greedy re-ranking: at each position k, pick item j maximizing score[j] + lam * (1 - exposure_share[group_j])
    m = len(score)
    chosen = []
    remaining = set(range(m))
    exposure = np.zeros(groups.max()+1)
    total_exp = 1e-12
    for k in range(m):
        best = None
        best_val = -1e18
        for j in list(remaining):
            g = groups[j]
            # potential exposure if placed here (alpha[k])
            # approximate exposure share after placing: (exposure[g] + alpha[k]) / (total_exp + alpha[k])
            share_g = (exposure[g] + alpha[k]) / (total_exp + alpha[k])
            val = score[j] + lam * (1 - share_g)
            if val > best_val:
                best_val = val
                best = j
        chosen.append(best)
        remaining.remove(best)
        g = groups[best]
        exposure[g] += alpha[k]
        total_exp += alpha[k]
    return np.array(chosen)


def rank_from_scores(score):
    return np.argsort(-score)


def welfare(u, ranking, alpha):
    # social welfare: sum_i sum_k alpha[k] * u[i, item_k]
    idx = ranking
    U = (u[:, idx] * alpha[None, :]).sum()
    return U


def exposure_entropy(ranking, groups, alpha, G):
    m = len(ranking)
    exp_by_g = np.zeros(G)
    for k, j in enumerate(ranking):
        g = groups[j]
        exp_by_g[g] += alpha[k]
    p = exp_by_g / exp_by_g.sum()
    H = entropy(p + 1e-12, base=np.e)
    # normalize by log(G)
    H_norm = H / math.log(G)
    # Herfindahl-Hirschman Index (HHI)
    HHI = (p**2).sum()
    return H, H_norm, HHI, p


def kendall_distance(r1, r2):
    # Convert rankings (orderings) to preference arrays positions
    m = len(r1)
    pos1 = np.empty(m, dtype=int)
    pos1[r1] = np.arange(m)
    pos2 = np.empty(m, dtype=int)
    pos2[r2] = np.arange(m)
    # Kendall tau correlation
    tau, _ = kendalltau(pos1, pos2)
    # sometimes nan if constant; handle
    if np.isnan(tau):
        tau = 0.0
    dist = (1 - tau) / 2
    return dist, tau


# --- Monte Carlo Simulation ---

def run_simulation(R=30, N=200, m=30, G=6, dims_list=(1,2), gammas=(0.0, 1.0, 2.0), rho=0.15, beta=4.0, lam_list=(0.0, 0.5)):
    results = []
    for dims in dims_list:
        for r in range(R):
            mu, X, groups, z = generate_environment(N=N, m=m, G=G, dims=dims, seed=10_000*r + dims)
            u = utilities(mu, X, sigma=0.5 if dims==1 else 0.7)
            rankings_ind = individual_rankings(u)
            borda_order, borda_points = borda_aggregate(rankings_ind)
            M = pairwise_matrix(rankings_ind, m)
            copeland_order, copeland_scores = copeland_ranking(M)
            cond_w = condorcet_winner(M)
            alpha = visibility_curve(m, rho=rho)

            # Neutral baseline (no virality amplification)
            e_neutral = platform_scores(u, z=np.ones_like(z), beta=beta, gamma=0.0)
            r_neutral = rank_from_scores(e_neutral)

            for gamma in gammas:
                e = platform_scores(u, z=z, beta=beta, gamma=gamma)
                r_engagement = rank_from_scores(e)

                for lam in lam_list:
                    if lam > 0:
                        r_platform = rerank_with_diversity(e, groups, alpha, lam=lam)
                    else:
                        r_platform = r_engagement

                    # Metrics
                    kw_borda, tau_borda = kendall_distance(r_platform, borda_order)
                    kw_copeland, tau_copeland = kendall_distance(r_platform, copeland_order)
                    w_platform = welfare(u, r_platform, alpha)
                    w_borda = welfare(u, borda_order, alpha)
                    w_copeland = welfare(u, copeland_order, alpha)
                    H, Hn, HHI, p = exposure_entropy(r_platform, groups, alpha, G)
                    H0, Hn0, HHI0, p0 = exposure_entropy(r_neutral, groups, alpha, G)
                    top1_share = p.max()

                    results.append({
                        'run': r,
                        'dims': dims,
                        'gamma': gamma,
                        'lambda': lam,
                        'kendall_borda': kw_borda,
                        'kendall_copeland': kw_copeland,
                        'tau_borda': tau_borda,
                        'tau_copeland': tau_copeland,
                        'welfare_platform': w_platform,
                        'welfare_borda': w_borda,
                        'welfare_copeland': w_copeland,
                        'entropy': H,
                        'entropy_norm': Hn,
                        'HHI': HHI,
                        'top1_exposure_share': top1_share,
                        'entropy_neutral_norm': Hn0,
                        'condorcet_exists': cond_w is not None,
                    })
    return pd.DataFrame(results)


# Run
sns.set(style='whitegrid')
df = run_simulation(R=40, N=200, m=30, G=6, dims_list=(1,2), gammas=(0.0, 1.0, 2.0), rho=0.15, beta=4.0, lam_list=(0.0, 0.6))

# Aggregate summaries
summary = df.groupby(['dims', 'gamma', 'lambda']).agg(
    kendall_borda_mean=('kendall_borda','mean'),
    kendall_copeland_mean=('kendall_copeland','mean'),
    entropy_norm_mean=('entropy_norm','mean'),
    welfare_gap_borda=('welfare_borda','mean'),
    welfare_platform_mean=('welfare_platform','mean'),
    top1_mean=('top1_exposure_share','mean'),
    condorcet_rate=('condorcet_exists','mean'),
    entropy_neutral_norm_mean=('entropy_neutral_norm','mean')
).reset_index()

# Create plots
fig1, ax1 = plt.subplots(figsize=(8,5))
for dims in (1,2):
    tmp = summary[(summary['lambda']==0.0) & (summary['dims']==dims)]
    ax1.plot(tmp['gamma'], tmp['kendall_borda_mean'], marker='o', label=f"Kendall vs Borda (dims={dims})")
ax1.set_title('Desalinhamento (Kendall) entre ranking da plataforma e Borda vs. γ (virality)')
ax1.set_xlabel('γ (amplificação de viralidade)')
ax1.set_ylabel('Distância de Kendall (0=igual, 1=oposto)')
ax1.legend()
fig1.tight_layout()
fig1_path = '/Users/albertosouza/Documents/UnB/Pesquisas/Escolha Social/fig1_kendall_borda_vs_gamma.png'
fig1.savefig(fig1_path, dpi=150)

# Entropy vs gamma for lambda=0 and lambda>0
fig2, ax2 = plt.subplots(figsize=(8,5))
for lam, style in [(0.0,'-o'), (0.6,'--s')]:
    tmp = summary[(summary['dims']==2) & (summary['lambda']==lam)]
    ax2.plot(tmp['gamma'], tmp['entropy_norm_mean'], style, label=f"λ={lam}")
# baseline neutral
tmp0 = summary[(summary['dims']==2)].groupby('gamma').entropy_neutral_norm_mean.mean().values
ax2.axhline(y=tmp0[0], color='gray', linestyle=':', label='Neutro (γ=0)')
ax2.set_title('Diversidade (Entropia Normalizada) vs. γ e λ (dims=2)')
ax2.set_xlabel('γ (amplificação de viralidade)')
ax2.set_ylabel('Entropia Normalizada (0-1)')
ax2.legend()
fig2.tight_layout()
fig2_path = '/Users/albertosouza/Documents/UnB/Pesquisas/Escolha Social/fig2_entropy_vs_gamma_lambda.png'
fig2.savefig(fig2_path, dpi=150)

# Trade-off: entropy vs Kendall (dims=2)
fig3, ax3 = plt.subplots(figsize=(6,5))
sub = summary[summary['dims']==2]
ax3.scatter(sub['kendall_borda_mean'], sub['entropy_norm_mean'], c=sub['gamma'], cmap='viridis', s=70)
for _, row in sub.iterrows():
    ax3.annotate(f"γ={row['gamma']},λ={row['lambda']}", (row['kendall_borda_mean'], row['entropy_norm_mean']), fontsize=7, alpha=0.7)
ax3.set_title('Fronteira: Diversidade vs. Desalinhamento (dims=2)')
ax3.set_xlabel('Distância de Kendall (plataforma vs Borda)')
ax3.set_ylabel('Entropia Normalizada')
fig3.tight_layout()
fig3_path = '/Users/albertosouza/Documents/UnB/Pesquisas/Escolha Social/fig3_tradeoff_entropy_kendall.png'
fig3.savefig(fig3_path, dpi=150)

# Save summary CSV
summary_path = '/Users/albertosouza/Documents/UnB/Pesquisas/Escolha Social/simulation_summary.csv'
df_path = '/Users/albertosouza/Documents/UnB/Pesquisas/Escolha Social/simulation_raw.csv'
summary.to_csv(summary_path, index=False)
df.to_csv(df_path, index=False)

(fig1_path, fig2_path, fig3_path, summary_path, df_path)
