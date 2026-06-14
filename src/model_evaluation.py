from sklearn.metrics import ( adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score )
import numpy as np
import pandas as pd

def purity_score(y_true, y_pred):

    contingency = pd.crosstab(y_pred, y_true)

    return ( np.sum(np.max(contingency.values, axis=1)) / np.sum(contingency.values))


def evaluate_clustering(y_true, K, n_classes):

    y_true = np.array(y_true).squeeze()

    metrics = {
        "ARI": adjusted_rand_score(y_true, K),
        "NMI": normalized_mutual_info_score(y_true, K),
        "Purity": purity_score(y_true, K),

        "Homogeneity": homogeneity_score(y_true, K),
        "Completeness": completeness_score(y_true, K),
        "VMeasure": v_measure_score(y_true, K),
        "FMI": fowlkes_mallows_score(y_true, K),

        "ART_Classes": n_classes,
    }

    return metrics