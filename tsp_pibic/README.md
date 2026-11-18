# TSP - 2-opt, Lin-Kernighan e AG

tsp.py -> algoritmos (2-opt, LK, AG e AG com 2-opt)
experimentos.py -> roda tudo 10x nas instancias da tsplib e salva os csv
graficos.py -> gera os graficos

pip install numpy numba pandas matplotlib python-tsp elkai
python experimentos.py
python graficos.py

gap = (custo - otimo) / otimo * 100

## VRP (varios veiculos, um deposito)

vrp.py -> agrupa os clientes (K-means, aglomerativo e DBSCAN), escolhe o k pela silhueta
e faz um TSP por grupo saindo do deposito
ag_agrupamento.py -> AG que escolhe os grupos usando a silhueta como aptidao

pip install scikit-learn
python vrp.py                              # 120 clientes sinteticos no RS
python vrp.py --tsp instancias/burma14.tsp
python vrp.py --csv pontos.csv --deposito -30.03,-51.23
python ag_agrupamento.py

o custo das rotas e sempre medido em km (haversine), mesmo quando o agrupamento usa euclidiana
saidas em resultados_vrp/
