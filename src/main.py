import art_model as art
import data_module as data
import model_evaluation as evaluation
import plots
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


def run_model(X, X_pca, y, p, feature_name):

    X_train, y_train, X_validation, y_validation, X_test, y_test = data.train_test_validation_split(X, y)

    X_pca_train, _, X_pca_validation, _, X_pca_test, _ =data.train_test_validation_split(X_pca, y)

    model = art.Art(X_train.shape[1], 1, p=p)

    Y_train, K_train = art.train_loop(model, X_train)
    Y_validation, K_validation = art.predict(model, X_validation)
    Y_test, K_test = art.predict(model, X_test)

    plots.compare_clusters(X_pca_test, y_test, K_test, title=f"{feature_name}_p_{p:.1f}")

    train_metrics = evaluation.evaluate_clustering(y_train, K_train, model.recognition_layer_size)
    validation_metrics = evaluation.evaluate_clustering(y_validation, K_validation, model.recognition_layer_size)
    test_metrics = evaluation.evaluate_clustering(y_test, K_test, model.recognition_layer_size)

    return train_metrics, validation_metrics, test_metrics


X, y = data.get_data()
X_pca = PCA(n_components=2).fit_transform(X)

feature_sets = {
    "Threshold": data.complement_coding(data.binary_thresholding(X)),
    "Quartiles": data.complement_coding(data.quartiles_binary_binning(X, q=4)),
    "Thermometer": data.complement_coding(data.thermometer_encoding(X, levels=4))
}

results = []
for feature_name, X_features in feature_sets.items():

    print(f"\nRunning {feature_name}")

    for p in np.arange(0.1, 1.0, 0.1):

        print("\np:", p)

        train_metrics, validation_metrics, test_metrics = run_model(X_features, X_pca, y, p, feature_name)

        row = {
            "Feature_Set": feature_name,
            "p": round(p, 2)
        }

        for metric_name, value in train_metrics.items():
            row[f"{metric_name}_Train"] = value

        for metric_name, value in validation_metrics.items():
            row[f"{metric_name}_Val"] = value

        for metric_name, value in test_metrics.items():
            row[f"{metric_name}_Test"] = value

        results.append(row)

results_df = pd.DataFrame(results)

results_df.to_csv("./results/art_experiment_results.csv",index=False)