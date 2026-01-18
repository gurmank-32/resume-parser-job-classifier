#train and predict
import pandas as pd

# Load dataset
df = pd.read_csv("data/resumes/resume_data.csv")  # path to Kaggle CSV

# Quick look
print(df.head())

df.columns = df.columns.str.strip()
print(df.columns)

df.rename(columns={'﻿job_position_name': 'job_position_name'}, inplace=True)
df['job_position_name'] = df['job_position_name'].astype(str).str.strip()
df['job_position_name'] = df['job_position_name'].replace(['', 'nan', 'None'], pd.NA)

df['job_position_name'].notna().sum()
df['job_position_name'].value_counts().head(10)

print(df['job_position_name'].value_counts())

# Check for nulls
print(df.isnull().sum())
df = df.dropna()  # drop missing rows

print(df.isnull().sum())



