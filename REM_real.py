import numpy as np
import csv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from REM import REM

#Ecoli
Data = np.genfromtxt('data/ecoli.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-4)

Cluster.fit(X)
1
8
0.15
aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['ecoli', 'aic', bndwk, aic_nc, aic_ari, aic_nmi])
  w.writerow(['ecoli', 'bic', bndwk, bic_nc, bic_ari, bic_nmi])
  w.writerow(['ecoli', 'icl', bndwk, icl_nc, icl_ari, icl_nmi])



#Iris
Data = np.genfromtxt('data/iris.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)
1
2
1

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['iris', 'aic', aic_nc, aic_ari, aic_nmi])
  w.writerow(['iris', 'bic', bic_nc, bic_ari, bic_nmi])
  w.writerow(['iris', 'icl', icl_nc, icl_ari, icl_nmi])



#Wine
Data = np.genfromtxt('data/wine.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)
 

Cluster.fit(X)
2
7

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['wine', 'aic', bndwk, aic_nc, aic_ari, aic_nmi, t2 - t1])
  w.writerow(['wine', 'bic', bndwk, bic_nc, bic_ari, bic_nmi, t2 - t1])
  w.writerow(['wine', 'icl', bndwk, icl_nc, icl_ari, icl_nmi, t2 - t1])



#Seeds
Data = np.genfromtxt('data/seeds.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)
1
1.2
2

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['seeds', 'aic', aic_nc, aic_ari, aic_nmi, t2 - t1])
  w.writerow(['seeds', 'bic', bic_nc, bic_ari, bic_nmi, t2 - t1])
  w.writerow(['seeds', 'icl', icl_nc, icl_ari, icl_nmi, t2 - t1])




#G2128
Data = np.genfromtxt('data/G2.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.sqrt(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)
1
0.0019
1200

aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['G2', 'aic', aic_nc, aic_ari, aic_nmi, t2 - t1])
  w.writerow(['G2', 'bic', bic_nc, bic_ari, bic_nmi, t2 - t1])
  w.writerow(['G2', 'icl', icl_nc, icl_ari, icl_nmi, t2 - t1])


#Satellite
Data = np.genfromtxt('data/satellite.csv', delimiter = ",")
X = Data[:, :-1]
y = Data[:, -1]

n_samples, n_features = X.shape

bndwk = int(np.floor(np.min((30, np.log(n_samples)))))
Cluster = REM.REM(covariance_type = "full", criteria = "all", bandwidth = bndwk, tol = 1e-5)

Cluster.fit(X)
1
0.045
40
aic_idx = np.argmin(Cluster.aics_)
bic_idx = np.argmin(Cluster.bics_)
icl_idx = np.argmin(Cluster.icls_)

aic_y = Cluster.mixtures[aic_idx].predict(X)
bic_y = Cluster.mixtures[bic_idx].predict(X)
icl_y = Cluster.mixtures[icl_idx].predict(X)

aic_nc = len(np.unique(aic_y))
bic_nc = len(np.unique(bic_y))
icl_nc = len(np.unique(icl_y))


aic_ari = adjusted_rand_score(aic_y.astype(int), y)
aic_nmi = normalized_mutual_info_score(aic_y.astype(int), y)
bic_ari = adjusted_rand_score(bic_y.astype(int), y)
bic_nmi = normalized_mutual_info_score(bic_y.astype(int), y)
icl_ari = adjusted_rand_score(icl_y.astype(int), y)
icl_nmi = normalized_mutual_info_score(icl_y.astype(int), y)

with open('REM_real.csv', 'a') as f:
  w = csv.writer(f)
  w.writerow(['satellite', 'aic', aic_nc, aic_ari, aic_nmi, t2 - t1])
  w.writerow(['satellite', 'bic', bic_nc, bic_ari, bic_nmi, t2 - t1])
  w.writerow(['satellite', 'icl', icl_nc, icl_ari, icl_nmi, t2 - t1])




