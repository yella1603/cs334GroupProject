from turtle import color
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/yellaleoniediekmann/cs334/selTrainDF.csv')

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
color = plt.get_cmap('RdYlGn')
color.set_bad('lightblue')
ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=color,
                cbar=True, square=True, linewidths=.5, 
                annot_kws={"size": 4}) 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
ax.tick_params(axis='both', which='major', labelsize=6)
plt.title('Correlation Matrix Heatmap')
plt.show()


