"""Week 11 exercises: k-means clustering and time complexity."""

from time import time

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


# Exercise 2 (setup): load diamonds and keep only the numeric columns.
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans(X, k):
   
    X = np.asarray(X, dtype=float)
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    model.fit(X)
    return model.cluster_centers_, model.labels_


def kmeans_diamonds(n, k):
    
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return float(np.mean(times))


# Bonus: binary search with a step counter.
step_count = 0


def bin_search(n):
    
    global step_count
    step_count = 0

    arr = np.arange(n)
    left = 0
    right = n - 1

    x = n - 1

    while left <= right:
        step_count += 1
        middle = left + (right - left) // 2

        # check if x is present at mid
        if arr[middle] == x:
            return middle

        # if x greater, ignore left half
        if arr[middle] < x:
            left = middle + 1

        # if x is smaller, ignore right half
        else:
            right = middle - 1

    # if we reach here, then element was not present
    return -1
