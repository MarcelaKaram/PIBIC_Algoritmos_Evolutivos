# AG para o agrupamento com a silhueta (haversine) como aptidao.
# individuo = k medoides, cada cliente vai pro medoide mais proximo
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit

from tsp import vizinhos
from vrp import args_dados, carrega, dist_haversine, kmeans, aglomerativo, rotear, silhueta, plota_rotas


@njit(cache=True)
def silhueta_nb(D, lab, k):
    # mesma conta do silhouette_score, mas em numba pq o AG chama milhares de vezes
    n = len(lab)
    cont = np.zeros(k, np.int64)
    for i in range(n):
        cont[lab[i]] += 1
    s = 0.0
    soma = np.empty(k)
    for i in range(n):
        if cont[lab[i]] == 1:
            continue  # grupo de 1 cliente conta como 0, igual ao sklearn
        soma[:] = 0.0
        for j in range(n):
            soma[lab[j]] += D[i, j]
        a = soma[lab[i]] / (cont[lab[i]] - 1)
        b = np.inf
        for c in range(k):
            if c != lab[i] and cont[c] > 0:
                b = min(b, soma[c] / cont[c])
        s += (b - a) / max(a, b)
    return s / n


@njit(cache=True)
def atribui(D, med):
    lab = np.empty(D.shape[0], np.int64)
    for i in range(D.shape[0]):
        m = 0
        for c in range(1, len(med)):
            if D[i, med[c]] < D[i, med[m]]:
                m = c
        lab[i] = m
    return lab


@njit(cache=True)
def ag_medoides(D, viz, k, seed, tam_pop=60, n_ger=300, taxa_mut=0.3, p_salto=0.1, elite=2):
    np.random.seed(seed)
    n = D.shape[0]
    pop = np.empty((tam_pop, k), np.int64)
    fit = np.empty(tam_pop)
    for i in range(tam_pop):
        pop[i] = np.random.permutation(n)[:k]
        fit[i] = silhueta_nb(D, atribui(D, pop[i]), k)
    hist = np.empty(n_ger)

    for g in range(n_ger):
        ordem = np.argsort(-fit)
        nova = np.empty_like(pop)
        nfit = np.empty_like(fit)
        for e in range(elite):
            nova[e] = pop[ordem[e]]
            nfit[e] = fit[ordem[e]]
        for i in range(elite, tam_pop):
            # torneio de 3
            p = np.empty(2, np.int64)
            for q in range(2):
                m = np.random.randint(0, tam_pop)
                for _ in range(2):
                    c = np.random.randint(0, tam_pop)
                    if fit[c] > fit[m]:
                        m = c
                p[q] = m
            # crossover: junta os medoides dos pais e sorteia k
            filho = np.unique(np.concatenate((pop[p[0]], pop[p[1]])))
            while len(filho) < k:
                filho = np.unique(np.append(filho, np.random.randint(0, n)))
            np.random.shuffle(filho)
            filho = filho[:k].copy()
            # mutacao: troca um medoide por um vizinho dele (as vezes por qualquer cliente)
            if np.random.random() < taxa_mut:
                j = np.random.randint(0, k)
                if np.random.random() < p_salto:
                    novo = np.random.randint(0, n)
                else:
                    novo = viz[filho[j], np.random.randint(0, viz.shape[1])]
                if not (filho == novo).any():
                    filho[j] = novo
            nova[i] = filho
            nfit[i] = silhueta_nb(D, atribui(D, filho), k)
        pop, fit = nova, nfit
        hist[g] = fit.max()

    b = np.argmax(fit)
    return pop[b], fit[b], hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args_dados(ap)
    ap.add_argument('--execucoes', type=int, default=5)
    ap.add_argument('--pop', type=int, default=60)
    ap.add_argument('--ger', type=int, default=300)
    a = ap.parse_args()
    pts, dep = carrega(a)

    Dh = dist_haversine(pts)
    viz = vizinhos(Dh, min(8, len(pts) - 1))
    ag_medoides(Dh, viz, 2, 0, 4, 2)  # compila o numba

    linhas, curvas, sols = [], {}, {}
    for k in range(2, min(a.kmax, len(pts) - 1) + 1):
        labs = {'K-means (euclidiana)': kmeans(pts, k),
                'Aglomerativo (haversine)': aglomerativo(Dh, k)}
        t = time.perf_counter()
        res = [ag_medoides(Dh, viz, k, s, a.pop, a.ger) for s in range(a.execucoes)]
        t_ag = (time.perf_counter() - t) / a.execucoes
        sils = [r[1] for r in res]
        med, _, curvas[k] = res[int(np.argmax(sils))]
        labs['AG medoides (haversine)'] = atribui(Dh, med)

        for nome, lab in labs.items():
            rotas, kms = rotear(pts, dep, lab, 'euclidiana' if 'euclid' in nome else 'haversine')
            l = dict(metodo=nome, k=k, silhueta_hav=silhueta(Dh, lab), km_total=kms.sum(),
                     km_max_rota=kms.max(), tam_min=np.bincount(lab).min(),
                     tam_max=np.bincount(lab).max())
            if nome.startswith('AG'):
                l.update(sil_ag_media=np.mean(sils), sil_ag_desvio=np.std(sils), t_ag=t_ag)
            linhas.append(l)
            sols[(nome, k)] = (lab, rotas, kms)
        sk, sa = silhueta(Dh, labs['K-means (euclidiana)']), silhueta(Dh, labs['Aglomerativo (haversine)'])
        print(f'k={k:2d}  AG {max(sils):.4f} (media {np.mean(sils):.4f})  '
              f'K-means {sk:.4f}  Aglom {sa:.4f}  [{t_ag:.1f}s/exec]')

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.saida, 'ag_resultados.csv'), index=False, float_format='%.4f')

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for nome, g in df.groupby('metodo', sort=False):
        ax[0].plot(g.k, g.silhueta_hav, 'o-', label=nome)
        ax[1].plot(g.k, g.km_total, 'o-', label=nome)
    for k, h in curvas.items():
        ax[2].plot(h, label=f'k={k}')
    ax[0].set(xlabel='k', ylabel='silhueta (haversine)', title='Aptidao por k')
    ax[1].set(xlabel='k', ylabel='distancia total (km)', title='Custo das rotas')
    ax[2].set(xlabel='geracao', ylabel='melhor silhueta', title='Convergencia do AG')
    for x in ax:
        x.grid(alpha=.3)
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, 'ag_silhueta_custo.png'), dpi=150)
    plt.close(fig)

    # rotas dos 3 metodos no k em que o AG teve a maior silhueta
    ag = df[df.metodo.str.startswith('AG')]
    kb = int(ag.loc[ag.silhueta_hav.idxmax(), 'k'])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for x, nome in zip(axs, df.metodo.unique()):
        lab, rotas, kms = sols[(nome, kb)]
        plota_rotas(x, pts, dep, rotas, f'{nome}\nk={kb}  sil={silhueta(Dh, lab):.3f}  {kms.sum():.0f} km')
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, 'ag_rotas.png'), dpi=150)
    plt.close(fig)

    pd.set_option('display.width', 200)
    print('\n' + df.round(3).to_string(index=False))
    print(f'\nmelhor k do AG: {kb}. arquivos em {a.saida}/')
