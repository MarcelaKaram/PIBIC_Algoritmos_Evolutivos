import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances

from tsp import ler_instancia, vizinhos, two_opt, lin_kernighan

R_TERRA = 6371.0  # km


def ler_tsplib_geo(arq):
    # tsplib GEO vem em graus.minutos (DDD.MM)
    p = ler_instancia(arq)
    return np.trunc(p) + (p - np.trunc(p)) * 100 / 60


def instancia_sintetica(n=120, seed=0):
    # clientes no RS: 80% em volta de algumas cidades, o resto espalhado
    rng = np.random.default_rng(seed)
    cidades = np.array([[-30.0, -51.2], [-29.2, -51.2], [-31.8, -52.3],
                        [-29.7, -53.8], [-28.3, -52.4], [-29.8, -57.1]])
    m = int(n * 0.8)
    perto = cidades[rng.integers(0, len(cidades), m)] + rng.normal(0, 0.35, (m, 2))
    longe = np.column_stack([rng.uniform(-33.0, -27.5, n - m),
                             rng.uniform(-57.5, -49.8, n - m)])
    return np.vstack([perto, longe])


def args_dados(ap):
    ap.add_argument('--tsp', help='tsplib com EDGE_WEIGHT_TYPE: GEO')
    ap.add_argument('--csv', help='csv com colunas lat,lon')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--deposito', help='lat,lon (padrao: centro dos clientes)')
    ap.add_argument('--kmax', type=int, default=10)
    ap.add_argument('--saida', default='resultados_vrp')


def carrega(a):
    if a.tsp:
        pts = ler_tsplib_geo(a.tsp)
    elif a.csv:
        pts = pd.read_csv(a.csv)[['lat', 'lon']].to_numpy(float)
    else:
        pts = instancia_sintetica(a.n, a.seed)
    dep = np.array([float(x) for x in a.deposito.split(',')]) if a.deposito else pts.mean(0)
    os.makedirs(a.saida, exist_ok=True)
    return pts, dep


def dist_haversine(a, b=None):
    # entra em graus (lat, lon), sai em km
    a = np.radians(a)
    return haversine_distances(a, a if b is None else np.radians(b)) * R_TERRA


def dist_euclid(a, b=None):
    # euclidiana direto nos graus
    return euclidean_distances(a, a if b is None else b)


def kmeans(pts, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(pts)


def aglomerativo(D, k):
    # nao tem metric='haversine', entao vai a matriz pronta. ward so aceita euclidiana
    return AgglomerativeClustering(k, metric='precomputed', linkage='average').fit_predict(D)


def dbscan_haversine(pts, eps_km, min_samples=4):
    lab = DBSCAN(eps=eps_km / R_TERRA, min_samples=min_samples, metric='haversine',
                 algorithm='ball_tree').fit_predict(np.radians(pts))
    # o ruido (-1) vai pro grupo do vizinho mais proximo, todo cliente tem que ser atendido
    ruido = lab == -1
    if ruido.all():
        return np.zeros(len(pts), int)
    if ruido.any():
        lab[ruido] = lab[~ruido][dist_haversine(pts[ruido], pts[~ruido]).argmin(1)]
    return lab


def silhueta(D, lab):
    k = len(np.unique(lab))
    if k < 2 or k >= len(lab):
        return np.nan
    return silhouette_score(D, lab, metric='precomputed')


def tsp_grupo(D):
    # no 0 = deposito. vizinho mais proximo, depois LK e 2-opt
    n = len(D)
    if n <= 3:
        return np.arange(n)
    r = [0]
    livre = set(range(1, n))
    while livre:
        j = min(livre, key=lambda v: D[r[-1], v])
        r.append(j)
        livre.remove(j)
    r = np.array(r, np.int64)
    r = two_opt(lin_kernighan(r, D, vizinhos(D, min(10, n - 1))), D)
    return np.roll(r, -int(np.where(r == 0)[0][0]))


def rotear(pts, dep, lab, metrica):
    # o TSP usa a metrica do agrupamento, mas o custo e sempre medido em km
    rotas, kms = [], []
    for g in np.unique(lab):
        idx = np.where(lab == g)[0]
        P = np.vstack([dep, pts[idx]])
        if metrica == 'haversine':
            D = np.rint(dist_haversine(P) * 1000).astype(np.int64)  # metros
        else:
            D = np.rint(dist_euclid(P) * 1e5).astype(np.int64)
        r = tsp_grupo(D)
        kms.append(dist_haversine(P)[r, np.roll(r, -1)].sum())
        rotas.append(np.concatenate([[-1], idx])[r])  # -1 = deposito
    return rotas, np.array(kms)


def plota_rotas(ax, pts, dep, rotas, titulo):
    cores = plt.cm.tab20.colors
    for i, r in enumerate(rotas):
        xy = np.array([dep if v < 0 else pts[v] for v in np.append(r, -1)])
        ax.plot(xy[:, 1], xy[:, 0], '-', color=cores[i % 20], lw=1)
        ax.scatter(xy[1:-1, 1], xy[1:-1, 0], s=12, color=cores[i % 20])
    ax.scatter(dep[1], dep[0], marker='*', s=250, c='k', zorder=5)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_aspect(1 / np.cos(np.radians(dep[0])))


def rodar(pts, dep, kmax, saida):
    Dh = dist_haversine(pts)
    De = dist_euclid(pts)
    ks = range(2, min(kmax, len(pts) - 1) + 1)
    linhas, sols = [], {}

    def registra(metodo, metrica, param, lab, t_agr):
        t = time.perf_counter()
        rotas, kms = rotear(pts, dep, lab, metrica)
        tam = np.bincount(lab)
        linhas.append(dict(metodo=metodo, metrica=metrica, param=param,
                           k=len(np.unique(lab)),
                           silhueta=silhueta(Dh if metrica == 'haversine' else De, lab),
                           silhueta_hav=silhueta(Dh, lab),
                           km_total=kms.sum(), km_max_rota=kms.max(), km_min_rota=kms.min(),
                           tam_min=tam.min(), tam_max=tam.max(),
                           t_agrup=t_agr, t_rota=time.perf_counter() - t))
        sols[(metodo, param)] = (lab, rotas, kms)

    for k in ks:
        for metodo, metrica, f in [('K-means', 'euclidiana', lambda: kmeans(pts, k)),
                                   ('Aglomerativo', 'euclidiana', lambda: aglomerativo(De, k)),
                                   ('Aglomerativo', 'haversine', lambda: aglomerativo(Dh, k))]:
            t = time.perf_counter()
            lab = f()
            registra(metodo, metrica, k, lab, time.perf_counter() - t)

    # no DBSCAN o k sai do eps. testa eps a partir da distancia ao 4o vizinho
    d4 = np.sort(Dh, axis=1)[:, 4]
    for eps in np.unique(np.round(np.percentile(d4, [50, 70, 80, 90, 95, 98]), 1)):
        t = time.perf_counter()
        lab = dbscan_haversine(pts, eps)
        if 2 <= len(np.unique(lab)) <= max(ks):
            registra('DBSCAN', 'haversine', f'eps={eps}km', lab, time.perf_counter() - t)

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(saida, 'vrp_resultados.csv'), index=False, float_format='%.4f')
    # melhor k de cada metodo pela silhueta na propria metrica
    melhores = df.loc[df.groupby(['metodo', 'metrica'], sort=False)['silhueta'].idxmax()]
    melhores.to_csv(os.path.join(saida, 'vrp_melhores.csv'), index=False, float_format='%.4f')
    return df, melhores, sols


def graficos(pts, dep, df, melhores, sols, saida):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (m, me), g in df[df.metodo != 'DBSCAN'].groupby(['metodo', 'metrica'], sort=False):
        ax[0].plot(g.k, g.silhueta, 'o-', label=f'{m} ({me})')
        ax[1].plot(g.k, g.km_total, 'o-', label=f'{m} ({me})')
    ax[0].set(xlabel='k (veiculos)', ylabel='silhueta', title='Escolha do k')
    ax[1].set(xlabel='k (veiculos)', ylabel='distancia total (km)', title='Custo das rotas')
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(saida, 'vrp_silhueta_custo.png'), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(melhores), figsize=(5 * len(melhores), 5), squeeze=False)
    for a, (_, l) in zip(axs[0], melhores.iterrows()):
        lab, rotas, kms = sols[(l.metodo, l.param)]
        plota_rotas(a, pts, dep, rotas, f'{l.metodo} ({l.metrica})\nk={l.k}  '
                    f'sil={l.silhueta:.3f}  {l.km_total:.0f} km')
    fig.tight_layout()
    fig.savefig(os.path.join(saida, 'vrp_rotas.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args_dados(ap)
    a = ap.parse_args()
    pts, dep = carrega(a)
    print(f'{len(pts)} clientes, deposito em ({dep[0]:.4f}, {dep[1]:.4f})')

    tsp_grupo(np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], np.int64))  # compila o numba
    df, melhores, sols = rodar(pts, dep, a.kmax, a.saida)
    graficos(pts, dep, df, melhores, sols, a.saida)

    cols = ['metodo', 'metrica', 'param', 'k', 'silhueta', 'silhueta_hav',
            'km_total', 'km_max_rota', 'tam_min', 'tam_max']
    pd.set_option('display.width', 200)
    print('\nmelhor k de cada metodo (pela silhueta):')
    print(melhores[cols].round(3).to_string(index=False))
    k = int(melhores.k.mode()[0])
    print('\nmesmo k para todos:')
    print(df[(df.k == k) & (df.metodo != 'DBSCAN')][cols].round(3).to_string(index=False))
    print(f'\narquivos em {a.saida}/')
