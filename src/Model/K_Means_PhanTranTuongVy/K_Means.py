# Chọn features mở rộng cho K-means
features = ['Age', 'Gender_encoded', 'Purchase Amount (USD)', 'Category_encoded', 'Review Rating', 'Previous Purchases',
            'Frequency of Purchases_encoded', 'Payment Method_encoded', 'Subscription Status_encoded',
            'Discount Applied_encoded', 'Season_encoded']
X_cluster = df[features]

# Tính inertia cho Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method Để Tìm Số Cluster Tối Ưu')
plt.xlabel('Số Cluster (k)')
plt.ylabel('Inertia')
plt.show()
