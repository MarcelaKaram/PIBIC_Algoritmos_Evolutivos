import numpy as np
from numba import njit


def ler_instancia(arq):
    pts = []
    with open(arq) as f:
        coords = False
        for l in f:
            l = l.strip()
            if l == "NODE_COORD_SECTION":
                coords = True
            elif l == "EOF":
                break
            elif coords and l:
                p = l.split()
                pts.append((float(p[1]), float(p[2])))
    return np.array(pts)


def dist_matrix(pts):
    # distancia euclidiana arredondada igual a tsplib (EUC_2D)
    d = pts[:, None, :] - pts[None, :, :]
    return np.floor(np.sqrt((d ** 2).sum(-1)) + 0.5).astype(np.int64)


def vizinhos(D, k=10):
    return np.argsort(D, axis=1)[:, 1:k + 1].astype(np.int64)


@njit(cache=True)
def custo(rota, D):
    n = len(rota)
    s = 0
    for i in range(n):
        s += D[rota[i], rota[(i + 1) % n]]
    return s


@njit(cache=True)
def two_opt(rota, D):
    r = rota.copy()
    n = len(r)
    melhora = True
    while melhora:
        melhora = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = r[i], r[i + 1]
                c, d = r[j], r[(j + 1) % n]
                # troca (a,b)(c,d) por (a,c)(b,d)
                if D[a, c] + D[b, d] < D[a, b] + D[c, d]:
                    r[i + 1:j + 1] = r[i + 1:j + 1][::-1].copy()
                    melhora = True
    return r


@njit(cache=True)
def inverte(r, pos, i, j):
    n = len(r)
    for _ in range(((j - i) % n + 1) // 2):
        r[i], r[j] = r[j], r[i]
        pos[r[i]] = i
        pos[r[j]] = j
        i = (i + 1) % n
        j = (j - 1) % n


@njit(cache=True)
def lin_kernighan(rota, D, viz, max_prof=50):
    r = rota.copy()
    n = len(r)
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[r[i]] = i
    marcado = np.zeros(n, np.int64)
    it = 0
    hi = np.empty(max_prof, np.int64)
    hj = np.empty(max_prof, np.int64)

    melhora = True
    while melhora:
        melhora = False
        for t1 in range(n):
            it += 1
            t2 = r[(pos[t1] + 1) % n]
            g = D[t1, t2]
            marcado[t1] = it
            marcado[t2] = it
            best_g, best_k, k = 0, 0, 0

            while k < max_prof:
                t3 = -1
                val = -1
                for v in viz[t2]:
                    if g - D[t2, v] <= 0:
                        break
                    if marcado[v] == it or v == t1:
                        continue
                    t4 = r[(pos[v] - 1) % n]
                    if t4 == t2:
                        continue
                    x = g - D[t2, v] + D[t4, v]
                    if x > val:
                        val = x
                        t3 = v
                if t3 == -1:
                    break

                t4 = r[(pos[t3] - 1) % n]
                hi[k], hj[k] = pos[t2], pos[t4]
                inverte(r, pos, pos[t2], pos[t4])
                k += 1
                marcado[t3] = it
                g = val
                t2 = t4
                # ganho se fechar o ciclo aqui
                if g - D[t2, t1] > best_g:
                    best_g = g - D[t2, t1]
                    best_k = k

            # volta as trocas que passaram do melhor ponto
            for p in range(k - 1, best_k - 1, -1):
                inverte(r, pos, hi[p], hj[p])
            if best_g > 0:
                melhora = True
    return r


@njit(cache=True)
def crossover_ox(p1, p2):
    n = len(p1)
    a, b = np.random.randint(0, n), np.random.randint(0, n)
    if a > b:
        a, b = b, a
    f = -np.ones(n, np.int64)
    usado = np.zeros(n, np.bool_)
    f[a:b + 1] = p1[a:b + 1]
    for c in p1[a:b + 1]:
        usado[c] = True
    k = (b + 1) % n
    for i in range(n):
        c = p2[(b + 1 + i) % n]
        if not usado[c]:
            f[k] = c
            k = (k + 1) % n
    return f


@njit(cache=True)
def torneio(fit, t=3):
    m = np.random.randint(0, len(fit))
    for _ in range(t - 1):
        c = np.random.randint(0, len(fit))
        if fit[c] < fit[m]:
            m = c
    return m


@njit(cache=True)
def ag(D, seed, tam_pop=100, n_ger=2000, taxa_mut=0.3, elite=2, busca_local=False):
    np.random.seed(seed)
    n = D.shape[0]
    pop = np.empty((tam_pop, n), np.int64)
    fit = np.empty(tam_pop, np.int64)
    for i in range(tam_pop):
        pop[i] = np.random.permutation(n)
        if busca_local:
            pop[i] = two_opt(pop[i], D)
        fit[i] = custo(pop[i], D)

    for _ in range(n_ger):
        ordem = np.argsort(fit)
        nova = np.empty_like(pop)
        for e in range(elite):
            nova[e] = pop[ordem[e]]
        for i in range(elite, tam_pop):
            filho = crossover_ox(pop[torneio(fit)], pop[torneio(fit)])
            if np.random.random() < taxa_mut:
                a, b = np.random.randint(0, n), np.random.randint(0, n)
                if a > b:
                    a, b = b, a
                filho[a:b + 1] = filho[a:b + 1][::-1].copy()
            if busca_local:
                filho = two_opt(filho, D)
            nova[i] = filho
        pop = nova
        for i in range(tam_pop):
            fit[i] = custo(pop[i], D)

    b = np.argmin(fit)
    return pop[b], fit[b]
