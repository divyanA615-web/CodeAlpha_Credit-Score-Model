import pandas as pd
df = pd.read_csv("D:/data science related/Internships/M_L/CodeAlpha_Credit-Score-Model/train_data.csv", nrows=1000)
print(df.dtypes)
print(df.head())
print("Null values:\n", df.isnull().sum())
print("Credit score unique values:\n", df['credit_score'].value_counts())
print("Credit history age unique values:\n", df['credit_history_age'].head(10))
