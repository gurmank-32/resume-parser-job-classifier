#train and predict
import pandas as pd

# Load dataset
df = pd.read_csv("data/resumes/resume_data.csv")  # path to Kaggle CSV

# Quick look
print(df.head())
# print(df['job_position_name'].value_counts())  # Make sure labels exist

# Check for nulls
print(df.isnull().sum())
df = df.dropna()  # drop missing rows
