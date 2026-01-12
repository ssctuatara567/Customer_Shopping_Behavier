# Top colors
top_colors = df['Color'].value_counts().head(10)
sns.barplot(x=top_colors.values, y=top_colors.index)
plt.title('Top Màu Sắc Phổ Biến')
plt.show()

# Color by gender
color_by_gender = df.groupby(['Gender', 'Color'])['Color'].count().unstack().fillna(0)
color_by_gender.plot(kind='bar', stacked=True)
plt.title('Màu Sắc Theo Giới Tính')
plt.show()
