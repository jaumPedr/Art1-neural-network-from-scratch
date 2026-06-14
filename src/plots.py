import pandas as pd
import matplotlib.pyplot as plt

def compare_clusters(X_2d, y_true, K, title):

    fig, ax = plt.subplots(1, 2, figsize=(14,6))

    ax[0].scatter( X_2d[:,0], X_2d[:,1], c=pd.factorize(y_true.squeeze())[0], cmap='tab10', s=10)
    ax[0].set_title("Classes Reais")

    ax[1].scatter(X_2d[:,0], X_2d[:,1], c=K, cmap='tab20', s=10)
    ax[1].set_title("Categorias ART")

    plt.tight_layout()

    plt.savefig(f"./results/{title}.png", dpi=300, bbox_inches="tight")
    plt.close()