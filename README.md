# Exp No.2 : Exploratory Data Analysis (EDA)

## Aim
To perform Exploratory Data Analysis (EDA) on the heart disease dataset to understand the data distribution, relationships, and key patterns useful for business insights and model building.

## Dataset
The dataset used for this experiment is `heartdisease.csv`. It contains medical and demographic information such as age, sex, cholesterol levels (`totChol`), `BMI`, blood pressure, and target values (`TenYearCHD`) indicating heart disease presence.

## Algorithm
1. Import required Python libraries (`numpy`, `pandas`, `seaborn`, `matplotlib`).
2. Load the dataset into a Pandas DataFrame.
3. Display the dataset structure, shape, and descriptive statistics.
4. Check for missing values and duplicate rows.
5. Examine correlations among numeric features using a correlation matrix and heatmap.
6. Visualize univariate distributions using histograms, boxplots, and countplots.
7. Analyze bivariate and multivariate relationships using scatter plots and pairplots.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Basic Exploration** | `df.info()` | Inspect dataset structure and data types |
| | `df.describe()` | Generate descriptive statistics |
| | `df.isnull().sum()` | Detect missing values in each column |
| | `df.duplicated().sum()` | Identify duplicate rows |
| | `df.corr()` | Find correlations between numeric variables |
| **Univariate Analysis** | `sns.histplot()` | Plot histogram (distribution of single variable) |
| | `sns.boxplot()` | Identify outliers using boxplot |
| | `sns.countplot()` | Count occurrences of categorical variables |
| **Bivariate Analysis** | `sns.scatterplot()` | Display relationship between two variables |
| | `sns.heatmap()` | Show correlation matrix visually |
| **Multivariate Analysis** | `sns.pairplot()` | Show pairwise relationships across variables |
| **Visualization Controls** | `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display plotted charts |

## Program & Output
# Step 1: Import Essential Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Optional: set a consistent plotting style
sns.set_theme(style="whitegrid")

# Step 2: Load the Dataset
# Keep heartdisease.csv in the same folder as this notebook.
df = pd.read_csv("heartdisease.csv")

print("Dataset loaded successfully.")
display(df.head())

# Step 3: Basic Exploration

# Inspect dataset structure and data types
print("\n--- Dataset Information ---")
df.info()

# Generate descriptive statistics for numerical columns
print("\n--- Descriptive Statistics ---")
display(df.describe())

# Check number of rows and columns
print("\n--- Dataset Shape ---")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

# Display first 10 rows
print("\n--- First 10 Rows ---")
display(df.head(10))

# Step 4: Check Data Quality

# Check for missing values in the dataset
print("\n--- Missing Values in Each Column ---")
print(df.isnull().sum())

# Check for duplicate rows
print("\n--- Number of Duplicate Rows ---")
print(df.duplicated().sum())

# Step 5: Correlation Analysis

# Calculate correlation between numeric columns
print("\n--- Correlation Matrix ---")
corr_matrix = df.corr(numeric_only=True)
display(corr_matrix)

# Visualize correlation using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.show()

# Step 6: Univariate Analysis

# Histogram for Age distribution
if "age" in df.columns:
    print("\n--- Histogram for Age Distribution ---")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"].dropna(), kde=True, bins=20, color="teal")
    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()
else:
    print("Column 'age' not found; age histogram skipped.")

# Boxplot for Cholesterol (to detect outliers)
if "totChol" in df.columns:
    print("\n--- Boxplot for Cholesterol ---")
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["totChol"])
    plt.title("Boxplot of Cholesterol Levels")
    plt.xlabel("Cholesterol")
    plt.show()
else:
    print("Column 'totChol' not found; cholesterol boxplot skipped.")

# Countplot for Gender (if available)
if "sex" in df.columns:
    print("\n--- Countplot for Gender ---")
    plt.figure(figsize=(5, 4))
    sns.countplot(x="sex", data=df)
    plt.title("Count of Gender")
    plt.xlabel("Sex (0 = Female, 1 = Male)")
    plt.ylabel("Count")
    plt.show()
else:
    print("Column 'sex' not found; gender countplot skipped.")

    # Step 7: Bivariate Analysis

# Scatter plot: Age vs Cholesterol
if {"age", "totChol"}.issubset(df.columns):
    print("\n--- Scatter Plot: Age vs Cholesterol ---")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="age", y="totChol", data=df,
        hue="TenYearCHD" if "TenYearCHD" in df.columns else None,
        palette="coolwarm" if "TenYearCHD" in df.columns else None
    )
    plt.title("Age vs Cholesterol (by Heart Disease Target)")
    plt.xlabel("Age")
    plt.ylabel("Cholesterol")
    plt.show()
else:
    print("Required columns 'age' and/or 'totChol' not found; scatter plot skipped.")

    # Step 8: Multivariate Analysis

# Pairplot for multiple relationships
selected_columns = [
    column for column in ["age", "totChol", "BMI", "TenYearCHD"]
    if column in df.columns
]

if len(selected_columns) >= 2:
    print("\n--- Pairplot for Multiple Relationships ---")
    sns.pairplot(
        df[selected_columns].dropna(),
        hue="TenYearCHD" if "TenYearCHD" in selected_columns else None,
        diag_kind="kde"
    )
    plt.suptitle("Pairwise Relationship between Variables", y=1.02)
    plt.show()
else:
    print("Not enough of the selected columns are present for a pairplot.")

    

## Result
