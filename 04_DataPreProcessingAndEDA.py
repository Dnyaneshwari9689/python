# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics as st

# Set Style
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Read Dataset
df = pd.read_csv("heart.csv")

print(df.head())

# Information of the Dataset
df.info()

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

# Data type of each column
print("Data Type for Each Column are:")
print(df.dtypes.value_counts())

print(df.describe())
print(df.head())

# Missing Values
n = df.columns[df.dtypes != 'object']

# Display all values
df[n]

# Dataframe size
size = df.size
print("Size of dataset is:", size)

# Dataframe shape
shape = df.shape
print("Shape of dataset is:\n", shape)

# All zeros / Missing values count
print(df.isnull().sum().sort_values(ascending=False))

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing Age values with median
df['age'] = df['age'].fillna(df['age'].median())

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

# --------------------------------
# Statistical Analysis
# --------------------------------

print("\nStatistical Summary:")
print(df.describe())

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# --------------------------------
# Data Visualization
# --------------------------------

# Histogram of Age
plt.figure(figsize=(8, 5))
plt.hist(df['age'], bins=20, edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram - Cholesterol
plt.figure(figsize=(8, 5))
plt.hist(df['chol'], bins=15, edgecolor='black')
plt.title('Histogram of Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# Boxplot - Age
plt.figure(figsize=(8, 5))
plt.boxplot(df['age'])
plt.title('Boxplot of Age')
plt.show()

# Boxplot - Cholesterol
plt.figure(figsize=(8, 5))
plt.boxplot(df['chol'])
plt.title('Boxplot of Cholesterol')
plt.show()

# Missing values of Age
df['age'] = df['age'].fillna(df['age'].median())
average = df['age'].mean()
print("Average age:", average)

# Selecting Feature Columns
print(df['age'])
print(df['sex'])
print(df['chol'])
print(df['oldpeak'])

# --------------------------------
# Splitting Data into Training
# and Testing Data
# --------------------------------

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:")
print(X_train.shape)

print("Testing Data Shape:")
print(X_test.shape)

print("Training Target Shape:")
print(y_train.shape)

print("Testing Target Shape:")
print(y_test.shape)
