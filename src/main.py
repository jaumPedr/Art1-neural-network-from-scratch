import art_model as art
import data_module as data
import model_evaluation as evaluation
import plots
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


#Train and evaluate an ART-1 model for a given feature representation and vigilance parameter.
def run_model(X, X_pca, y, p, feature_name):
    
    #Split encoded features into training, validation, and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = data.train_test_validation_split(X, y)
    
    # Split PCA representation using the same proportions
    X_pca_train, _, X_pca_validation, _, X_pca_test, _ =data.train_test_validation_split(X_pca, y)

    #ART1 model
    model = art.Art(X_train.shape[1], 1, p=p)

    #Train the model
    Y_train, K_train = art.train_loop(model, X_train)
    
     #Evaluate on validation and test sets
    Y_validation, K_validation = art.predict(model, X_validation)
    Y_test, K_test = art.predict(model, X_test)

    #Generate cluster visualization
    plots.compare_clusters(X_pca_test, y_test, K_test, title=f"{feature_name}_p_{p:.1f}")

    #Compute clustering metrics
    train_metrics = evaluation.evaluate_clustering(y_train, K_train, model.recognition_layer_size)
    validation_metrics = evaluation.evaluate_clustering(y_validation, K_validation, model.recognition_layer_size)
    test_metrics = evaluation.evaluate_clustering(y_test, K_test, model.recognition_layer_size)

    return train_metrics, validation_metrics, test_metrics

#Load dataset
X, y = data.get_data()

#Compute a 2D PCA projection for visualization
X_pca = PCA(n_components=2).fit_transform(X)

#Feature representations
feature_sets = {
    "Threshold": data.complement_coding(data.binary_thresholding(X)),
    "Quartiles": data.complement_coding(data.quartiles_binary_binning(X, q=4)),
    "Thermometer": data.complement_coding(data.thermometer_encoding(X, levels=4))
}

#Run experiments
results = []
for feature_name, X_features in feature_sets.items():

    print(f"\nRunning {feature_name}")

    #Evaluate multiple P values
    for p in np.arange(0.1, 1.0, 0.1):

        print("\np:", p)

        train_metrics, validation_metrics, test_metrics = run_model(X_features, X_pca, y, p, feature_name)

        row = {
            "Feature_Set": feature_name,
            "p": round(p, 2)
        }

        # Store metrics
        for metric_name, value in train_metrics.items():
            row[f"{metric_name}_Train"] = value

        for metric_name, value in validation_metrics.items():
            row[f"{metric_name}_Val"] = value

        for metric_name, value in test_metrics.items():
            row[f"{metric_name}_Test"] = value

        results.append(row)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("./results/art_experiment_results.csv",index=False)