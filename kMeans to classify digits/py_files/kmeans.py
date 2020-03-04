# @Author: Pipe galera
# @Date:   28-02-2020
# @Email:  pipegalera@gmail.com
# @Last modified by:   Pipe galera
# @Last modified time: 04-03-2020
%reset
# Packages
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_digits

#load datasets
digits = load_digits()
data = scale(digits.data) # Scale down data to speed up computation

# Implementation
y = digits.target
k = 10

def bench_k_means(estimator, name, data): # From sklearn
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf_kmeans = KMeans(n_clusters = k, init = "k-means++", n_init = 10)
bench_k_means(clf_kmeans, "Different mesurements of accuracy:", data)
