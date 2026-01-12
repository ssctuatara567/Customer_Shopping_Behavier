from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Hàm để evaluate và visualize chi tiết cho một k
def evaluate_kmeans_detailed(k, X):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Silhouette score tổng
    sil_score = silhouette_score(X, labels)
    print(f"Silhouette Score tổng cho k={k}: {sil_score:.4f}")

    # Silhouette visualizer: Plot silhouette cho từng sample
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_sil_values = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(k):
        ith_cluster_sil_values = sample_sil_values[labels == i]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.set_title(f'Silhouette Plot Cho k={k}')
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=sil_score, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

    # Visualize clusters với PCA và centers
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(centers)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centers_pca[:,0], centers_pca[:,1], c='red', marker='x', s=200, label='Centers')
    plt.title(f'PCA Clusters Và Centers Cho k={k}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.legend()
    plt.show()

    # Mean per cluster dưới dạng table
    df_temp = df.copy()
    df_temp['Cluster'] = labels
    cluster_means = df_temp.groupby('Cluster').mean(numeric_only=True)
    print(f"Table Mean Per Cluster Cho k={k}:")
    display(cluster_means)  # Sử dụng display để in table đẹp trong Colab
    print("\n")

# Thử với k=2 và k=4
evaluate_kmeans_detailed(2, X_cluster)
evaluate_kmeans_detailed(4, X_cluster)
