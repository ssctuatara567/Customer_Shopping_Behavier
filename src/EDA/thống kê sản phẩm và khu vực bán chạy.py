# Khu vực bán chạy nhất
location_sales = df.groupby('Location')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
sns.barplot(x=location_sales.values, y=location_sales.index)
plt.title('Doanh Thu Theo Vị Trí')
plt.show()

# Top sản phẩm theo location
top_products_by_loc = df.groupby(['Location', 'Item Purchased'])['Purchase Amount (USD)'].sum().unstack().fillna(0)
print(top_products_by_loc)
