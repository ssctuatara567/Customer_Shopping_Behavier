
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import networkx as nx

print("🚀 Bắt đầu xây dựng hệ thống gợi ý tối ưu...")

# ==========================================
# PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU CHUẨN CHO KNN
# ==========================================

# 1. Chọn Features (Bỏ Location vì nhiễu, giữ lại các biến hành vi quan trọng)
features_knn = ['Age', 'Gender', 'Season', 'Subscription Status', 'Previous Purchases', 'Frequency of Purchases']
item_col = 'Item Purchased'

# 2. Xử lý dữ liệu
df_knn = df[features_knn].copy()

# A. Xử lý Frequency (Ordinal Encoding - Gán số theo độ lớn tần suất)
freq_map = {
    'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26,
    'Monthly': 12, 'Quarterly': 4, 'Every 3 Months': 4, 'Annually': 1
}
df_knn['Freq_Score'] = df_knn['Frequency of Purchases'].map(freq_map).fillna(1)
df_knn.drop('Frequency of Purchases', axis=1, inplace=True)

# B. One-Hot Encoding (Chuẩn cho biến phân loại)
df_knn_encoded = pd.get_dummies(df_knn, columns=['Gender', 'Season', 'Subscription Status'], drop_first=True)
final_feature_cols = df_knn_encoded.columns.tolist()

# 3. Chuẩn hóa (StandardScaler)
scaler = StandardScaler()
X_knn_scaled = scaler.fit_transform(df_knn_encoded)
y_items = df[item_col].values

# 4. Huấn luyện KNN
knn = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='auto')
knn.fit(X_knn_scaled)

# --- KHAI BÁO MÀU SẮC TOÀN CỤC
cat_colors = {
    'Clothing': '#3498db',    # Xanh dương
    'Accessories': '#2ecc71', # Xanh lá
    'Footwear': '#e74c3c',    # Đỏ
    'Outerwear': '#f1c40f'    # Vàng
}

print("✅ Đã huấn luyện xong mô hình KNN.")
print(f"📊 Số lượng đặc trưng đầu vào: {len(final_feature_cols)} features")
print("\n" + "="*50 + "\n")

# ==========================================
# PHẦN 2: HÀM GỢI Ý & VISUALIZATION CÁ NHÂN
# ==========================================

def visualize_recommendations(customer_input):
    """
    Input: Dictionary chứa thông tin khách hàng
    """
    # --- A. Xử lý Input ---
    try:
        input_df = pd.DataFrame([customer_input])
        input_df['Freq_Score'] = input_df['Frequency of Purchases'].map(freq_map).fillna(1)
        input_df.drop('Frequency of Purchases', axis=1, inplace=True)

        input_encoded = pd.get_dummies(input_df, columns=['Gender', 'Season', 'Subscription Status'], drop_first=True)
        input_encoded = input_encoded.reindex(columns=final_feature_cols, fill_value=0)

        input_scaled = scaler.transform(input_encoded)

    except Exception as e:
        print(f"❌ Lỗi xử lý dữ liệu: {e}")
        return

    # --- B. Tìm hàng xóm ---
    distances, indices = knn.kneighbors(input_scaled)
    neighbor_indices = indices[0]

    neighbor_items = y_items[neighbor_indices]
    neighbor_df = df.iloc[neighbor_indices].copy()
    neighbor_df['Similarity'] = (1 - distances[0]).round(4)

    # --- C. Vẽ Dashboard ---
    top_items = pd.Series(neighbor_items).value_counts().head(5)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.5]})

    ax[0].axis('off')
    info_text = f"👤 KHÁCH HÀNG MỤC TIÊU\n{'='*30}\n"
    for k, v in customer_input.items():
        info_text += f"• {k}: {v}\n"
    info_text += f"\n🔍 KẾT QUẢ KNN:\n• Đã tìm thấy 50 'hàng xóm' gần nhất."

    ax[0].text(0.05, 0.95, info_text, fontsize=12, va='top', family='monospace', linespacing=1.6,
               bbox=dict(boxstyle="round,pad=1", fc="#f0f8ff", ec="navy", alpha=0.5))

    colors = sns.color_palette("viridis", len(top_items))
    bars = ax[1].barh(top_items.index, top_items.values, color=colors)
    ax[1].invert_yaxis()
    ax[1].set_title('TOP 5 SẢN PHẨM GỢI Ý', fontsize=15, fontweight='bold', color='#2E8B57')
    for bar in bars:
        ax[1].text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2, f'{int(bar.get_width())}', va='center')

    plt.tight_layout()
    plt.show()

    # --- D. Vẽ Mạng lưới Cá nhân (Star Graph) ---
    print(f"\n🕸️ MẠNG LƯỚI TƯƠNG ĐỒNG (YOU & TOP 15 HÀNG XÓM)\n{'='*60}")
    G = nx.Graph()
    G.add_node("YOU", color='red', size=2500, label="YOU")

    top_15 = neighbor_df.head(15)

    for idx, row in top_15.iterrows():
        cat = row.get('Category', 'Unknown')
        color = cat_colors.get(cat, 'gray') # Lấy màu từ biến toàn cục
        G.add_node(idx, color=color, size=600, label=row['Item Purchased'])
        G.add_edge("YOU", idx, weight=row['Similarity']*2)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.4, seed=42)

    node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    node_sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.4)

    labels = nx.get_node_attributes(G, 'label')
    pos_labels = {k: (v[0], v[1]-0.08) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos_labels, labels=labels, font_size=9)
    nx.draw_networkx_labels(G, pos, labels={"YOU":"YOU"}, font_color='white', font_weight='bold')

    plt.axis('off')
    plt.show()

# ==========================================
# PHẦN 3: VISUALIZATION TOÀN CỤC (GLOBAL)
# ==========================================

print("🌍 Đang tạo biểu đồ trực quan toàn cục (Global Visualization)...")

# 1. t-SNE
print("   -> Chạy t-SNE (có thể mất 10-20s)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_emb = tsne.fit_transform(X_knn_scaled)

df_plot = pd.DataFrame(X_emb, columns=['X', 'Y'])
df_plot['Gender'] = df['Gender']
df_plot['Category'] = df['Category']

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_plot, x='X', y='Y', hue='Gender', palette='coolwarm', alpha=0.6)
plt.title('Không gian khách hàng (Phân theo Giới tính)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_plot, x='X', y='Y', hue='Category', palette='tab10', alpha=0.5)
plt.title('Không gian khách hàng (Phân theo Loại hàng)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Network Graph (Sample 200 người)
print("   -> Vẽ mạng lưới mẫu (Sample 200 nodes)...")
sample_idx = np.random.choice(X_knn_scaled.shape[0], 200, replace=False)
X_sample = X_knn_scaled[sample_idx]

knn_sample = NearestNeighbors(n_neighbors=3, metric='cosine').fit(X_sample)
dists, neighs = knn_sample.kneighbors(X_sample)

G_global = nx.Graph()
for i, real_idx in enumerate(sample_idx):
    cat = df.iloc[real_idx]['Category']
    G_global.add_node(i, category=cat)
    for n_idx in neighs[i]:
        if i != n_idx:
            G_global.add_edge(i, n_idx)

plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G_global, k=0.15, seed=42)


node_colors = [cat_colors.get(nx.get_node_attributes(G_global, 'category')[n], 'gray') for n in G_global.nodes()]

nx.draw_networkx_nodes(G_global, pos, node_size=80, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G_global, pos, width=0.3, alpha=0.3)
plt.title("Mạng lưới tương đồng của 200 khách hàng ngẫu nhiên", fontsize=15)
plt.axis('off')
plt.show()

# ==========================================
# CHẠY THỬ NGHIỆM
# ==========================================
print("\n🧪 DEMO: Chạy thử với 1 khách hàng...")
my_customer = {
    'Age': 28,
    'Gender': 'Male',
    'Season': 'Spring',
    'Location': 'California', # Cột này sẽ bị bỏ qua tự động
    'Subscription Status': 'No',
    'Previous Purchases': 2,
    'Frequency of Purchases': 'Annually'
}
visualize_recommendations(my_customer)
