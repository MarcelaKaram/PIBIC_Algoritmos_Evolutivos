# qual metrica de avaliacao do agrupamento acompanha melhor o custo real das rotas?
# testa silhueta, davies-bouldin, calinski-harabasz e uma proposta nova (custo_mst)
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from vrp import (args_dados, carrega, dist_haversine, dist_euclid, kmeans, aglomerativo, rotear,
                 silhueta, tsp_grupo, R_TERRA)
from ag_agrupamento import ag_medoides, atribui
from mst_mis import agrupa_mst, conjuntos_mis, sementes_mis
from tsp import vizinhos


def custo_mst(pts, dep, lab):
    # proposta: soma das arvores geradoras minimas de cada grupo junto com o deposito.
    # a rota de um grupo fica entre 1x e 2x a sua arvore, entao isso estima o custo
    # das rotas sem resolver nenhum TSP, e ja conta a ida e volta ao deposito
    total = 0.0
    for g in np.unique(lab):
        total += minimum_spanning_tree(dist_haversine(np.vstack([dep, pts[lab == g]]))).sum()
    return total


# para cada metrica: funcao e se maior e melhor
METRICAS = {
    'silhueta': (lambda pts, dep, lab, Dh, xy: silhueta(Dh, lab), True),
    'davies_bouldin': (lambda pts, dep, lab, Dh, xy: davies_bouldin_score(xy, lab), False),
    'calinski_harabasz': (lambda pts, dep, lab, Dh, xy: calinski_harabasz_score(xy, lab), True),
    'custo_mst': (lambda pts, dep, lab, Dh, xy: custo_mst(pts, dep, lab), False),
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args_dados(ap)
    a = ap.parse_args()
    pts, dep = carrega(a)
    Dh, De = dist_haversine(pts), dist_euclid(pts)
    # davies-bouldin e calinski-harabasz precisam de coordenadas: projeta lat/lon em km
    xy = np.radians(pts) * R_TERRA * np.array([1, np.cos(np.radians(dep[0]))])
    viz = vizinhos(Dh, min(8, len(pts) - 1))
    mis = conjuntos_mis(Dh, np.linspace(10, Dh.max() / 2, 60))
    tsp_grupo(np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], np.int64))  # compila o numba

    linhas, tempos = [], {m: 0.0 for m in METRICAS}
    tempos['rotas (TSP)'] = 0.0
    for k in range(2, min(a.kmax, len(pts) - 1) + 1):
        sem, _ = sementes_mis(mis, k)
        labs = {'K-means': kmeans(pts, k),
                'Aglomerativo euclidiana': aglomerativo(De, k),
                'Aglomerativo haversine': aglomerativo(Dh, k),
                'Corte da MST': agrupa_mst(Dh, k),
                'Medoides MIS': atribui(Dh, sem),
                'K-means iniciado no MIS': KMeans(k, init=pts[sem], n_init=1).fit_predict(pts),
                'AG medoides': atribui(Dh, ag_medoides(Dh, viz, k, 0)[0])}
        for nome, lab in labs.items():
            l = dict(metodo=nome, k=k)
            for m, (f, _) in METRICAS.items():
                t = time.perf_counter()
                l[m] = f(pts, dep, lab, Dh, xy)
                tempos[m] += time.perf_counter() - t
            # rotas sempre com haversine, pra comparar so o agrupamento
            t = time.perf_counter()
            l['km_total'] = rotear(pts, dep, lab, 'haversine')[1].sum()
            tempos['rotas (TSP)'] += time.perf_counter() - t
            linhas.append(l)
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.saida, 'metricas_resultados.csv'), index=False, float_format='%.4f')

    # concordancia com o custo real (spearman com sinal ajustado: 1 = ordena igual ao km)
    res = []
    for m, (_, maior_melhor) in METRICAS.items():
        s = -1 if maior_melhor else 1
        geral = s * spearmanr(df[m], df.km_total)[0]
        por_k = np.mean([s * spearmanr(g[m], g.km_total)[0] for _, g in df.groupby('k')])
        # em quantos k a metrica aponta o mesmo metodo que tem o menor km
        acertos = sum((g[m].idxmax() if maior_melhor else g[m].idxmin()) == g.km_total.idxmin()
                      for _, g in df.groupby('k'))
        res.append(dict(metrica=m, concordancia_geral=geral, concordancia_mesmo_k=por_k,
                        acertos_melhor_metodo=f'{acertos}/{df.k.nunique()}',
                        tempo_s=tempos[m]))
    res.append(dict(metrica='rotas (TSP)', tempo_s=tempos['rotas (TSP)']))
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(a.saida, 'metricas_concordancia.csv'), index=False, float_format='%.4f')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for x, m in zip(ax, ['silhueta', 'custo_mst']):
        for nome, g in df.groupby('metodo', sort=False):
            x.scatter(g[m], g.km_total, s=18, label=nome)
        x.set(xlabel=m + (' (km)' if m == 'custo_mst' else ''), ylabel='km total das rotas')
        x.grid(alpha=.3)
    ax[0].set_title('Silhueta x custo real')
    ax[1].set_title('Custo da MST (proposta) x custo real')
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, 'metricas.png'), dpi=150)
    plt.close(fig)

    pd.set_option('display.width', 200)
    print(res.round(3).to_string(index=False))
    print(f'\narquivos em {a.saida}/')
