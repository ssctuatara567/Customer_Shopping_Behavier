# Top sizes
top_sizes = df['Size'].value_counts()
sns.barplot(x=top_sizes.values, y=top_sizes.index)
plt.title('Phân Bố Kích Cỡ')
plt.show()

# Size by category
size_by_category = df.groupby(['Category', 'Size'])['Size'].count().unstack().fillna(0)
size_by_category.plot(kind='bar', stacked=True)
plt.title('Kích Cỡ Theo Category')
plt.show()
