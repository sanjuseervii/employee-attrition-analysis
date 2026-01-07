import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('Employee.csv')
print(df.shape)
print("="*60)
print(df.columns)
print("="*60)
print(df.head())
print("="*60)
print(df.info()) 
print("="*60)
print(df.isnull().sum())
print("="*60)
print(df.describe())
numeric_values = df.select_dtypes(include=['int64', 'float64'])
for cols in numeric_values:
    plt.figure(figsize=(6,4))
    sns.histplot(x=df[cols],kde=True,bins=20)
    plt.show()
sns.boxplot(x=df['JoiningYear'])
plt.show()
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
cat_columns=['Education','City','EverBenched','Gender']
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.countplot(x=df[var])
num=1
for cols in cat_columns:
    plotting(cols,num)
    num=num+1
plt.tight_layout()
plt.show()
for cols in cat_columns:
    df.groupby(cols)['LeaveOrNot'].value_counts().unstack().plot(kind='bar')
plt.tight_layout()
#plt.show()

print(df.columns)
df_cleaned=df.copy()


# 1. Identify categorical and numeric columns
categorical_cols = ['Education', 'City', 'Gender', 'EverBenched', 'LeaveOrNot']  # category wale
numeric_cols = ['PaymentTier', 'Age', 'ExperienceInCurrentDomain', 'JoiningYear']  # numeric wale

# 2. Convert categorical columns to numeric (One-Hot Encoding)
df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# 3. Scale numeric columns
scaler = StandardScaler()
df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])

# 4. Check final dataframe
print(df_cleaned.head())
# True/False → 0/1 for all bool columns automatically
bool_cols = df_cleaned.select_dtypes(include='bool').columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)
print(df_cleaned.head())
#applying stats 
# 1. Correlation matrix
corr_matrix = df_cleaned.corr(method='pearson')

# 2. Print correlation with target column (for FE insight)
# Assuming target column is 'LeaveOrNot' ya jo bhi predict karna hai
target = 'LeaveOrNot_1'
print(corr_matrix[target].sort_values(ascending=False))

# 3. Optional: Heatmap for visualization
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()
print(df_cleaned.columns)
import pandas as pd
from scipy.stats import chi2_contingency

target = 'LeaveOrNot_1'

categorical_features = [
    'City_Pune',
    'City_New Delhi',
    'Education_Masters',
    'Education_PHD',
    'Gender_Male',
    'EverBenched_Yes'
]

results = []

for feature in categorical_features:
    contingency_table = pd.crosstab(df_cleaned[feature], df_cleaned[target])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    results.append({
        'Feature': feature,
        'Chi2': chi2,
        'p-value': p,
        'Decision': 'ACCEPT' if p < 0.05 else 'REJECT (drop feature)'
    })

chi_square_df = pd.DataFrame(results).sort_values(by='p-value')
print(chi_square_df)
count_left=len(df[df['LeaveOrNot']==1])
print('count of employees left',count_left)
female_left=df_cleaned[(df_cleaned['Gender_Male']==0) & (df_cleaned['LeaveOrNot_1']==1)]
print('no of female employees left:',len(female_left))
male_left=df_cleaned[(df_cleaned['Gender_Male']==1) & (df_cleaned['LeaveOrNot_1']==1)]
print('no of male employees left:',len(male_left))
city_wise_attrition=df.groupby('City')['LeaveOrNot'].value_counts().sort_values(ascending=False)
print(city_wise_attrition)
education_wise_attrition=df.groupby('Education')['LeaveOrNot'].value_counts()
print(education_wise_attrition)
df_cleaned.to_csv('employee_data.csv',index=False)