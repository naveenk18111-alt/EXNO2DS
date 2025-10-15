# EXNO2DS
# DATE: 15-10-2025
# NAME : NAVEENKUMAR V
# REG NO: 25016071
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("titanic_dataset.csv")
df
```
<img width="1359" height="652" alt="Screenshot 2025-09-18 143159" src="https://github.com/user-attachments/assets/50a1292a-1f33-404c-a46a-72b48c3a4bfa" />


```
df.info()
```
<img width="740" height="464" alt="Screenshot 2025-09-18 143211" src="https://github.com/user-attachments/assets/befa6bf7-ddcc-4ac2-ad3c-68b7edf754c2" />


```
df.describe()
```
<img width="939" height="373" alt="Screenshot 2025-09-18 143230" src="https://github.com/user-attachments/assets/ac5d04bc-b0c0-408c-ac5e-c1622e5b9ab8" />


```
df.dtypes
```
<img width="529" height="337" alt="Screenshot 2025-09-18 143316" src="https://github.com/user-attachments/assets/9eba3115-22b3-4c30-8b21-8f2cb94e86d7" />

```
df.shape
```
<img width="386" height="100" alt="Screenshot 2025-09-18 143335" src="https://github.com/user-attachments/assets/e44aa347-9316-4559-88d1-081e58b559bf" />

```
df.value_counts()
```
<img width="1403" height="596" alt="Screenshot 2025-09-18 143407" src="https://github.com/user-attachments/assets/2f346629-def2-43e4-8e70-6ee2bb245f66" />


```
df['Age'].value_counts()
```
<img width="611" height="312" alt="Screenshot 2025-09-18 143543" src="https://github.com/user-attachments/assets/538f5c3c-8835-4c89-ac71-bc3dd6999a3a" />


```
df_set.index("PassengerId",inplace=True)
df
```
<img width="1328" height="560" alt="Screenshot 2025-09-18 143639" src="https://github.com/user-attachments/assets/159f826b-0b13-4b91-8464-1b22d6c43d55" />

```
df.nunique
```
<img width="587" height="315" alt="Screenshot 2025-09-18 143712" src="https://github.com/user-attachments/assets/640bfed0-dc48-4ecc-85af-ac59989112df" />

```
sns.countplot(data=df,x='Survived')
```
<img width="1280" height="656" alt="Screenshot 2025-09-18 143732" src="https://github.com/user-attachments/assets/fa4859a1-2311-41ca-910c-af0b3d2cffca" />

```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
<img width="1333" height="585" alt="Screenshot 2025-09-18 143758" src="https://github.com/user-attachments/assets/9ba722ad-3067-41cd-ab7e-73a965e3687c" />


```
sns.catplot(x='Gender',col='Survived',kind="count",data=df,height=5,aspect=1)
```
<img width="1192" height="716" alt="Screenshot 2025-09-18 143855" src="https://github.com/user-attachments/assets/9095fde1-6189-45a2-80b5-3d62acbfd0ca" />

```
df.boxplot(column='Age',by='Survived')
df
```
<img width="887" height="678" alt="Screenshot 2025-09-18 143916" src="https://github.com/user-attachments/assets/5f3d61f4-efba-48a3-8a67-e209d1663654" />

```
sns.scatterplot(x=df['Age'],y=df['Fare'])
```
<img width="948" height="641" alt="Screenshot 2025-09-18 143935" src="https://github.com/user-attachments/assets/80c1aac8-6b2a-43ae-ba76-235316230be3" />

```
fig, axl=plt.subplots(figsize=(8,5))
plt=sns.boxplot(ax=axl,x='Pclass',y='Age',hue='Gender',data=df)
```
<img width="1043" height="641" alt="Screenshot 2025-09-18 144011" src="https://github.com/user-attachments/assets/4cfb9e76-7078-4471-b2d3-7d436562ef72" />

```
plt=sns.boxplot(x='Pclass',y='Age',hue='Gender',data=df)
```
<img width="914" height="618" alt="Screenshot 2025-09-18 144050" src="https://github.com/user-attachments/assets/2021fd0a-e5a5-4a40-9f79-e8995cb96f25" />
```
import seaborn as sns
sns.catplot(x='Pclass',y="Age",hue="Gender",col="Survived",kind="box",data=df)
```
<img width="1340" height="684" alt="Screenshot 2025-09-18 144215" src="https://github.com/user-attachments/assets/4631dffa-fc50-4de9-98e2-91987c16e1df" />
```
sns.catplot(data=df,col="Survived",x="Gender",hue='Pclass',kind="count")
```
<img width="1345" height="656" alt="Screenshot 2025-09-18 144255" src="https://github.com/user-attachments/assets/474d7ec4-642f-4584-9d49-cccc6defded3" />

```
corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True)

```
<img width="1096" height="632" alt="Screenshot 2025-09-18 144356" src="https://github.com/user-attachments/assets/5fa0ade4-18c9-4795-b7ed-f6f6f7996937" />


```
corr=df.select_dtypes(include=np.number).corr()
sns.heatmap(corr,annot=True)
```
<img width="837" height="633" alt="Screenshot 2025-09-18 144420" src="https://github.com/user-attachments/assets/a52967ff-8569-4665-82cc-e34efedd35ee" />


# RESULT
Exploratory Data Analysis on the given data set is successful
