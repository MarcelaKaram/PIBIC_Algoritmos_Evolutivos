import time
import numpy as np
import pandas as pd
import elkai
from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search
from tsp import *

# otimos conhecidos da tsplib
otimos = {'eil51': 426, 'berlin52': 7542, 'st70': 675, 'eil76': 538,
          'kroA100': 21282, 'ch150': 6528, 'kroA200': 29368}
N_RUNS = 10


def lkh(D):
    r = elkai.DistanceMatrix(D.tolist()).solve_tsp(runs=1)
    return custo(np.array(r[:-1]), D)


# roda uma vez antes so pra compilar o numba e nao atrapalhar o tempo
D = dist_matrix(ler_instancia('instancias/eil51.tsp'))
two_opt(np.arange(51), D)
lin_kernighan(np.arange(51), D, vizinhos(D))
ag(D, 0, 10, 2)
ag(D, 0, 4, 1, busca_local=True)

dados = []
for inst, otimo in otimos.items():
    print(inst)
    D = dist_matrix(ler_instancia(f'instancias/{inst}.tsp'))
    V = vizinhos(D)
    n = len(D)
    for s in range(N_RUNS):
        r0 = np.random.default_rng(s).permutation(n)  # mesma rota inicial pra todos
        metodos = {
            '2-opt': lambda: custo(two_opt(r0, D), D),
            'LK': lambda: custo(lin_kernighan(r0, D, V), D),
            'AG': lambda: ag(D, s, 100, 10000)[1],
            'AG+2opt': lambda: ag(D, s, 30, 50, busca_local=True)[1],
            '2-opt python-tsp': lambda: solve_tsp_local_search(D.astype(float), x0=list(r0))[1],
            'LK python-tsp': lambda: solve_tsp_lin_kernighan(D.astype(float), x0=list(r0))[1],
            'LKH elkai': lambda: lkh(D),
        }
        for nome, f in metodos.items():
            t = time.perf_counter()
            c = f()
            t = time.perf_counter() - t
            dados.append([inst, n, nome, s, int(c), otimo, 100 * (c - otimo) / otimo, t])

df = pd.DataFrame(dados, columns=['instancia', 'n', 'metodo', 'seed', 'custo', 'otimo', 'gap', 'tempo'])
df.to_csv('resultados_brutos.csv', index=False)

resumo = df.groupby(['instancia', 'n', 'metodo'], sort=False).agg(
    otimo=('otimo', 'first'), melhor=('custo', 'min'), media=('custo', 'mean'),
    desvio=('custo', 'std'), gap_medio=('gap', 'mean'), gap_min=('gap', 'min'),
    tempo_medio=('tempo', 'mean')).round(3).reset_index()
resumo.to_csv('resultados_resumo.csv', index=False)
print(resumo)
