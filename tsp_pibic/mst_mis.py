# arvore geradora minima (agrupar e rota inicial) e conjunto independente maximo (sementes)
import argparse
import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import maximum_independent_set
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans

from tsp import custo
from vrp import (args_dados, carrega, dist_haversine, kmeans, rotear, silhueta, plota_rotas,
                 tsp_grupo, vizinho_mais_proximo)
from ag_agrupamento import atribui


def agrupa_mst(D, k):
    # corta as k-1 arestas mais longas da arvore e sobram k pedacos (= single linkage)
    T = minimum_spanning_tree(D).toarray()
    i, j = np.nonzero(T)
    for e in np.argsort(T[i, j])[::-1][:k - 1]:
        T[i[e], j[e]] = 0
    return connected_components(T, directed=False)[1]


def rota_mst(D):
    # percorre a arvore em profundidade a partir do deposito (garante custo <= 2x o otimo)
    G = nx.from_numpy_array(minimum_spanning_tree(D).toarray())
    return np.array(list(nx.dfs_preorder_nodes(G, 0)), np.int64)


def conjuntos_mis(D, raios):
    # grafo com aresta entre clientes a menos de r km. um conjunto independente e um
    # grupo de clientes todos a pelo menos r km uns dos outros
    res = {}
    for r in raios:
        G = nx.from_numpy_array(((D < r) & (D > 0)).astype(int))
        res[r] = sorted(maximum_independent_set(G))
    return res


def sementes_mis(mis, k):
    # maior raio que ainda deixa k clientes afastados entre si
    r = max(r for r, S in mis.items() if len(S) >= k)
    return np.array(mis[r][:k]), r


def compara_rota_inicial(pts, dep, lab):
    # mesma rotina do tsp_grupo, so muda a rota de partida
    linhas = []
    for g in np.unique(lab):
        P = np.vstack([dep, pts[lab == g]])
        if len(P) <= 3:
            continue
        D = np.rint(dist_haversine(P) * 1000).astype(np.int64)
        for nome, r0 in [('vizinho mais proximo', vizinho_mais_proximo(D)), ('arvore (MST)', rota_mst(D))]:
            r = tsp_grupo(D, r0)
            linhas.append(dict(grupo=g, n=len(P), inicio=nome, km_inicial=custo(r0, D) / 1000,
                               km_final=custo(r, D) / 1000))
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args_dados(ap)
    a = ap.parse_args()
    pts, dep = carrega(a)
    Dh = dist_haversine(pts)
    tsp_grupo(np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], np.int64))  # compila o numba

    mis = conjuntos_mis(Dh, np.linspace(10, Dh.max() / 2, 60))
    linhas, sols = [], {}
    for k in range(2, min(a.kmax, len(pts) - 1) + 1):
        sem, raio = sementes_mis(mis, k)
        labs = {'K-means': kmeans(pts, k),
                'Corte da MST': agrupa_mst(Dh, k),
                'Medoides MIS': atribui(Dh, sem),
                'K-means iniciado no MIS': KMeans(k, init=pts[sem], n_init=1).fit_predict(pts)}
        for nome, lab in labs.items():
            rotas, kms = rotear(pts, dep, lab, 'euclidiana' if nome.startswith('K-means') else 'haversine')
            tam = np.bincount(lab)
            linhas.append(dict(metodo=nome, k=k, silhueta_hav=silhueta(Dh, lab), km_total=kms.sum(),
                               km_max_rota=kms.max(), tam_min=tam.min(), tam_max=tam.max(),
                               raio_mis_km=raio if 'MIS' in nome else np.nan))
            sols[(nome, k)] = (lab, rotas, kms)
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.saida, 'mst_mis_resultados.csv'), index=False, float_format='%.4f')

    # rota inicial: um veiculo so (todos os clientes) e os grupos do K-means com k=6
    ini = pd.concat([compara_rota_inicial(pts, dep, np.zeros(len(pts), int)).assign(caso='k=1'),
                     compara_rota_inicial(pts, dep, sols[('K-means', 6)][0]).assign(caso='K-means k=6')])
    ini.to_csv(os.path.join(a.saida, 'mst_rota_inicial.csv'), index=False, float_format='%.2f')

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    T = minimum_spanning_tree(Dh).toarray()
    i, j = np.nonzero(T)
    cortes = set(np.argsort(T[i, j])[::-1][:5])
    for e in range(len(i)):
        c, w = ('tab:red', 2) if e in cortes else ('0.6', 0.8)
        ax[0].plot(pts[[i[e], j[e]], 1], pts[[i[e], j[e]], 0], color=c, lw=w)
    ax[0].scatter(pts[:, 1], pts[:, 0], s=8, c='k')
    ax[0].set_title('Arvore geradora minima\n(em vermelho, as 5 arestas cortadas para k=6)', fontsize=10)
    ax[0].set_aspect(1 / np.cos(np.radians(dep[0])))
    for x, nome in zip(ax[1:], ['Corte da MST', 'Medoides MIS']):
        lab, rotas, kms = sols[(nome, 6)]
        plota_rotas(x, pts, dep, rotas, f'{nome}, k=6\nsil={silhueta(Dh, lab):.3f}  {kms.sum():.0f} km')
    sem, raio = sementes_mis(mis, 6)
    ax[2].scatter(pts[sem, 1], pts[sem, 0], s=120, facecolors='none', edgecolors='k', lw=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, 'mst_mis.png'), dpi=150)
    plt.close(fig)

    pd.set_option('display.width', 200)
    print(df.pivot(index='k', columns='metodo', values='km_total').round(0).to_string())
    print('\nsilhueta haversine:')
    print(df.pivot(index='k', columns='metodo', values='silhueta_hav').round(3).to_string())
    print('\ntamanho do menor grupo:')
    print(df.pivot(index='k', columns='metodo', values='tam_min').to_string())
    print('\nrota inicial (km somados):')
    print(ini.groupby(['caso', 'inicio'])[['km_inicial', 'km_final']].sum().round(1).to_string())
    print(f'\narquivos em {a.saida}/')
