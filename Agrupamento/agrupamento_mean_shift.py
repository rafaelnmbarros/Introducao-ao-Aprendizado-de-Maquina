"""
====================================================
Demonstração do algoritmo de agrupamento mean-shift
====================================================

Referências:

    1.https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-download-auto-examples-cluster-plot-mean-shift-py
    
    2. Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# %%
# Geração de dados amostrais
#==============================================================================
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=50000, centers=centers, cluster_std=0.425)

# %%
# Computar o agrupamento com o MeanShift
#==============================================================================

# A seguinte largura de banda pode ser detectada automaticamente usando
largura_faixa = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=largura_faixa, bin_seeding=True)
ms.fit(X)
niveis = ms.labels_
centros_grupos = ms.cluster_centers_

rotulos_unicos = np.unique(niveis)
n_grupos_ = len(rotulos_unicos)

print("Número de grupos estimado : %d" % n_grupos_)

# %%
# Exibição dos resultados
#==============================================================================
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_grupos_), colors):
    my_members = niveis == k
    centro_grupo = centros_grupos[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + ",", alpha=0.25)
    plt.plot(
        centro_grupo[0],
        centro_grupo[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Número de grupos estimado: %d" % n_grupos_)
plt.show()
