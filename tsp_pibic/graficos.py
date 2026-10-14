import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsp import *

res = pd.read_csv('resultados_resumo.csv')
metodos = ['2-opt', '2-opt python-tsp', 'LK python-tsp', 'AG', 'LK', 'AG+2opt', 'LKH elkai']
insts = res.instancia.unique()
x = np.arange(len(insts))
w = 0.12

plt.figure(figsize=(13, 5.5))
for i, m in enumerate(metodos):
    gaps = [res[(res.instancia == k) & (res.metodo == m)].gap_medio.values[0] for k in insts]
    plt.bar(x + (i - 3) * w, gaps, w, label=m, edgecolor='k', lw=0.4)
plt.xticks(x, [f'{k}\n(n={res[res.instancia == k].n.iloc[0]})' for k in insts])
plt.ylabel('gap medio (%)')
plt.title('Gap em relacao ao otimo - media de 10 execucoes')
plt.legend(ncol=4, fontsize=8.5)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_gap.png', dpi=150)

# rotas da kroA100
pts = ler_instancia('instancias/kroA100.tsp')
D = dist_matrix(pts)
r0 = np.random.default_rng(0).permutation(len(pts))
rotas = {
    'inicial (aleatoria)': r0,
    '2-opt': two_opt(r0, D),
    'Lin-Kernighan': lin_kernighan(r0, D, vizinhos(D)),
    'AG + 2-opt': ag(D, 0, 30, 50, busca_local=True)[0],
}
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for a, (nome, r) in zip(ax, rotas.items()):
    p = pts[np.append(r, r[0])]
    a.plot(p[:, 0], p[:, 1], lw=0.9)
    a.scatter(pts[:, 0], pts[:, 1], s=9, c='k', zorder=3)
    c = custo(r, D)
    a.set_title(f'{nome}\ncusto = {c} (gap {100 * (c - 21282) / 21282:.1f}%)', fontsize=10)
    a.set_xticks([])
    a.set_yticks([])
plt.suptitle('kroA100 (otimo = 21282)')
plt.tight_layout()
plt.savefig('rotas_kroA100.png', dpi=150)
