# Group by Season and Category
season_category = df.groupby(['Season', 'Category'])['Purchase Amount (USD)'].sum().unstack().fillna(0)
season_category.plot(kind='bar', stacked=True)
plt.title('Doanh Thu Theo Mùa Và Category')
plt.show()

# Count
season_category_count = df.groupby(['Season', 'Category'])['Category'].count().unstack().fillna(0)
season_category_count.plot(kind='bar', stacked=True)
plt.title('Số Lượng Mua Theo Mùa Và Category')
plt.show()
