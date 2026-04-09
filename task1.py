import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Titanic-Dataset.csv')


#1.Import the dataset and explore basic info (nu ls, data types)

print("------ FIRST 5 ROWS ------")
print(df.head())

print("\n------ DATASET INFO ------")
print(df.info())

print("\n------ MISSING VALUE ------")
print(df.isnull().sum())


#2.Handle missing values using mean/median/imputation.

df['Age'] = df['Age'].fillna(df['Age'].median(), inplace = True)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

df.drop(columns=['Cabin'], inplace=True)

print("\n----- AFTER CLEANING MISSING VALUES -----")
print(df.isnull().sum())


#.Convert categorical features into numerical using encoding

df['Sex'] = df['Sex'].map({'male':0, 'female':0})

df = pd.get_dummies(df, columns=['Embarked'], drop_first = True)

print(df.head())


#Normalize/standardize the numerical features.
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\n----- AFTER NORMALIZATION -----")
print(df[['Age', 'Fare']].head())


#Visualize outliers using boxplots and remove them.
plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare Before Outlier Removal")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= Q1 - 1.5 * IQR) &
        (df['Fare'] <= Q3 + 1.5 * IQR)]

plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare After Outlier Removal")
plt.show()

print("\n----- FINAL DATASET SHAPE -----")
print(df.shape)

df.to_csv('cleaned_titanic.csv', index=False)

print("\n✅ Data Cleaning Completed Successfully!")