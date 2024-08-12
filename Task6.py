'''
Description:
Apply advanced statistical and analytical
methods to solve complex problems.
Responsibility:
1. Implement time series analysis for
forecasting trends and seasonality.
2. Perform sentiment analysis or text mining on
unstructured data.
Explore clustering or classification
techniques for segmentation and pattern
recognition.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:\Python\Datasets\Task6DiabetesDataset.csv")
#Checking columns names
print(df.columns.values)
print("Shape of Dataset is : ",df.shape)

#Dropping Null Values
df.dropna(inplace=True)
print("Shape of Dataset is : ",df.shape)
df.isna().sum()
#Printing info and description of dataset
df.info()
df.describe()


#Visualization
#Ques.1 How many persons have Diabetes and  how many do not have diabetes

df.target.value_counts()
#plotting results using bar chart
df.target.value_counts().plot(kind='bar',color=['b','r'],alpha=0.6)
plt.title("Diabetes",fontsize=25)
plt.xlabel("0=No Diabetes 1=Diabetes",fontsize=15)
plt.ylabel("Number of Persons",fontsize=15)
plt.show()

# Ques2.  How many male and female are in the dataset
df.sex.value_counts()
#Plotting results
df.sex.value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.legend(['Female','Male'])
plt.legend("Male Female Ratio")
plt.show()

#Ques.3 Persons of which sex has the most Positive cases of Diabetes
pd.crosstab(df.target,df.sex)
sns.countplot(x='target',hue='sex',data=df)
plt.title("Diabetes Frequency for Sex")
plt.xlabel("0=No Diabetes 1=Diabetes")
plt.ylabel("Numbers")
plt.legend(["Male","Female"],loc=2)
plt.show()

#Ques 4.What BMI range has the highest number of positive cases of diabetes?

# Creating BMI ranges
bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
df['bmi_range'] = pd.cut(df['bmi'], bins=bins, labels=labels, right=False)

# Crosstab to see the distribution
pd.crosstab(df['target'], df['bmi_range'])

# Plotting the results
plt.figure(figsize=(10, 6))
sns.countplot(x='bmi_range', hue='target', data=df)
plt.title("Diabetes Frequency by BMI Range")
plt.xlabel("BMI Range")
plt.ylabel("Count")
plt.legend(["No Diabetes", "Diabetes"], loc='upper right')
plt.show()

#Ques.5 What blood pressure (bp) range has the highest number of positive cases of diabetes?

# Creating blood pressure (bp) ranges
bins = [0, 80, 120, 140, 200]
labels = ['Low', 'Normal', 'Pre-high', 'High']
df['bp_range'] = pd.cut(df['bp'], bins=bins, labels=labels, right=False)

# Crosstab to see the distribution
bp_distribution = pd.crosstab(df['target'], df['bp_range'])

# Plotting the results
plt.figure(figsize=(10, 6))
sns.countplot(x='bp_range', hue='target', data=df)
plt.title("Diabetes Frequency by Blood Pressure Range")
plt.xlabel("Blood Pressure Range")
plt.ylabel("Count")
plt.legend(["No Diabetes", "Diabetes"], loc='upper right')
plt.show()
