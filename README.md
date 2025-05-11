
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np
from scipy.stats import ttest_ind

# 1. Load and Explore the Dataset (with error handling)
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

print("--- Data Exploration ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 2. More Detailed Exploration and Advanced Data Analysis
print("\n--- Correlation Matrix ---")
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Iris Features")
plt.show()

print("\n--- Outlier Detection ---")
plt.figure(figsize=(12, 6))
df.drop('target', axis=1).boxplot()
plt.title("Boxplot of Iris Features")
plt.ylabel("cm")
plt.show()

print("\n--- Advanced Statistics ---")
print("Skewness:\n", df.skew(numeric_only=True))
print("\nKurtosis:\n", df.kurtosis(numeric_only=True))
print("\nPercentiles:\n", df.quantile([0.25, 0.5, 0.75]))

# Hypothesis Testing
setosa_sepal_length = df[df['species'] == 'setosa']['sepal length (cm)']
versicolor_sepal_length = df[df['species'] == 'versicolor']['sepal length (cm)']
t_statistic, p_value = ttest_ind(setosa_sepal_length, versicolor_sepal_length)
print("\n--- Hypothesis Testing ---")
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in sepal lengths.")
else:
    print("Fail to reject the null hypothesis: No significant difference in sepal lengths.")

# 3. Enhanced Visualizations

# a) Histogram
print("\n--- Histogram ---")
df['sepal length (cm)'].hist(bins=20)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# b) Bar Chart
print("\n--- Bar Chart ---")
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar')
plt.title("Average Petal Length by Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# c) Pair Plot
print("\n--- Pair Plot ---")
sns.pairplot(df, hue='species')
plt.suptitle("Pair Plot of Iris Features by Species", y=1.02)
plt.show()

# d) Violin Plot
print("\n--- Violin Plot ---")
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal length (cm)', data=df)
plt.title("Violin Plot of Sepal Length by Species")
plt.ylabel("Sepal Length (cm)")
plt.show()

# e) Scatter Plot
print("\n--- Customized Scatter Plot ---")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='viridis')
plt.title("Sepal Length vs. Petal Length (with Species)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# 4. Summary of Findings
print("\n--- Summary of Findings ---")
print("""
* The correlation matrix shows strong positive correlations between petal length and petal width.
* Box plots reveal potential outliers (although the Iris dataset is generally clean).
* Skewness and kurtosis provide insights into the distribution shapes.
* The t-test suggests a statistically significant difference in sepal lengths between Setosa and Versicolor.
* Histograms and bar charts help explore feature distributions and differences between species.
* Pair plots and violin plots provide a clear visual summary of species separation.
* Line charts were not included as the dataset does not have a time component.
""")

