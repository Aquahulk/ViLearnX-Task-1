
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files


uploaded = files.upload()


df = pd.read_csv("train.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the dataset structure and summary statistics
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())



# Drop the 'Cabin' column due to a high percentage of missing values
df.drop(columns=['Cabin'], inplace=True)

# Check for duplicates and remove them
df.drop_duplicates(inplace=True)

# Display the cleaned dataset info
print("\nCleaned Dataset Info:")
print(df.info())

# Summary statistics for numerical features
print("\nSummary Statistics for Numerical Features:")
print(df.describe())


print("\nFare Distribution:")
print(df['Fare'].describe())

# Correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualization
sns.set(style="whitegrid")


# Heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
