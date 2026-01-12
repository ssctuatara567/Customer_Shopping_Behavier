# --- TASK DỰ ĐOÁN CHI TIÊU (IMPROVED VERSION) ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("🚀 Đang phân tích bài toán Dự đoán Chi tiêu...")

# 1. FEATURE SELECTION & ENGINEERING
# Chọn Features đầu vào cụ thể (tránh lấy nhầm cột rác)
features_spending = [
    'Age', 'Gender', 'Category', 'Season', 'Subscription Status',
    'Review Rating', 'Previous Purchases'
]
target_col = 'Purchase Amount (USD)'

# Tạo bản sao
df_reg = df[features_spending + [target_col]].copy()

# One-Hot Encoding (Tốt hơn Label Encoding cho Regression)
# Giúp máy hiểu 'Category_Clothing' là 1 đặc điểm riêng biệt
df_reg = pd.get_dummies(df_reg, columns=['Gender', 'Category', 'Season', 'Subscription Status'], drop_first=True)

# 2. KIỂM TRA TƯƠNG QUAN (DIAGNOSIS)
# Bước này để trả lời tại sao R2 thấp
correlation = df_reg.corr()[target_col].sort_values(ascending=False)
print("\n📊 Mức độ tương quan với số tiền chi tiêu (Purchase Amount):")
print(correlation.head(10)) # In ra những yếu tố ảnh hưởng nhất

# Vẽ Heatmap
plt.figure(figsize=(10, 8))
# Lấy Top 10 features quan trọng nhất để vẽ cho đỡ rối
top_features = correlation.abs().nlargest(10).index
sns.heatmap(df_reg[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Biểu đồ Tương quan (Heatmap)")
plt.show()

# 3. CHUẨN BỊ DATA TRAIN
X = df_reg.drop(target_col, axis=1)
y = df_reg[target_col]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (Gradient Boosting chạy tốt hơn khi data cùng scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. HUẤN LUYỆN MODEL (Gradient Boosting)
# GBR thường mạnh hơn RF trong việc tìm các mối liên hệ tuyến tính nhỏ
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gbr.fit(X_train_scaled, y_train)

# 5. ĐÁNH GIÁ
y_pred = gbr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print(f"🏆 KẾT QUẢ MÔ HÌNH (GRADIENT BOOSTING):")
print(f"• MSE (Sai số bình phương trung bình): {mse:.2f}")
print(f"• R2 Score (Độ phù hợp): {r2:.4f}")
print("="*40)

# 6. TRỰC QUAN HÓA: DỰ ĐOÁN VS THỰC TẾ
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='purple')
# Vẽ đường chéo đỏ (Dự đoán hoàn hảo)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Thực tế (Actual Amount)')
plt.ylabel('Dự đoán (Predicted Amount)')
plt.title('Biểu đồ phân tán: Giá trị Thực tế vs Dự đoán')
plt.show()
