import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#A1
df = pd.read_excel("Lab Session Data.xlsx",sheet_name="Purchase data")
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values
print(f"The dimensonality of the vector space of X is {np.linalg.matrix_rank(X)}")
print(X.shape)
c = np.linalg.pinv(X)@y
print(f"The cost of each product is {c}")
print(y.shape)

#A2
customers = df["Customer"].values
i = 0
for price, customer in zip(y, customers):
    if price >= 200:
        print(f"{customer} : RICH")
    else:
        print(f"{customer} : POOR")
y_class = (y >= 200).astype(int)
w = np.linalg.pinv(X) @ y_class
pred = (X @ w >= 0.5).astype(int)
print("Classifier Predictions:", pred)

#A3
df = pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
D = df["Price"].values
print("#1")
print(f"Mean of the prices is {np.mean(D)}")
print(f"Variance of the prices is {np.var(D)}")

def mean_own(column):
    sum = 0
    for value in column:
        sum += value
    return sum/len(column)

def var_own(column):
    total = 0
    for value in column:
        total += (value - mean_own(column))**2
    return total/(len(column)-1)

wed_rows = df["Day"] == "Wed"
apr_rows = df["Month"] == "Apr"
E = df.loc[wed_rows, "Price"]
K = df.loc[apr_rows, "Price"]
print(mean_own(D))
print(var_own(D))
print("Wednesday Mean and Variances")
print(mean_own(E))
print(var_own(E))
print("April Mean and Variance")
print(mean_own(K))
print(var_own(K))
chg_rows = df["Chg%"].values
print("The probabilty of making a Profit")
print(mean_own(chg_rows > 0))
print("The probabilty of making a Loss")
print(mean_own(chg_rows < 0))
wed_chg_rows = df.loc[df["Day"] == "Wed", "Chg%"]
print(mean_own(wed_chg_rows > 0))
x = df["Day"]
y = df["Chg%"]
plt.scatter(x,y)
plt.show()

#A4
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
num = df.select_dtypes(include=np.number)
mean = num.mean()
std = num.std()
outliers = ((num < (mean - 3*std)) | (num > (mean + 3*std))).sum()
print("Outliers count:\n", outliers)
print("Mean:\n", mean)
print("Variance:\n", num.var())

#A5
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
vec1_1 = df["Kidhome"] == 1
vec1_2 = df["Kidhome"] == 0
vec2_1 = df["Teenhome"] == 1
vec2_2 = df["Teenhome"] == 0
vec1 = df["Kidhome"]
vec2 = df["Teenhome"]
print(np.shape(vec1_1))
f11 = ((vec1_1) & (vec2_1)).sum()
f10 = ((vec1_1) & (vec2_2)).sum()
f01 = ((vec1_2) & (vec2_1)).sum()
f00 = ((vec1_2) & (vec2_2)).sum()
JC = f11/(f01 + f10 + f11)
SMC = (f11 + f00)/(f00 + f01 + f10 +f11)

#A6
X = df.loc[:19, ["Kidhome", "Teenhome"]].values
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#A7
def jacard(a,b):
    f11 = np.sum((a==1)&(b==1))
    f10 = np.sum((a==1)&(b==0))
    f01 = np.sum((a==0)&(b==1))
    f00 = np.sum((a==0)&(b==0))
    return f11 / (f11 + f10 + f01)

def smc(a,b):
    f11 = np.sum((a==1)&(b==1))
    f10 = np.sum((a==1)&(b==0))
    f01 = np.sum((a==0)&(b==1))
    f00 = np.sum((a==0)&(b==0))
    return (f11 + f00)/ (f11 + f10 + f01 + f00)

n = X.shape[0]
JC = np.zeros((n,n))
SMC = np.zeros((n,n))
COS = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        JC[i,j] = jacard(X[i], X[j])
        SMC[i,j] = smc(X[i], X[j])
        COS[i,j] = cosine(X[i], X[j])

#A8
num = df.select_dtypes(include=np.number)
cat = df.select_dtypes(exclude=np.number)
for c in num.columns:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].median())
for c in cat.columns:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].mode()[0])

#A9
cols = ["Income","Recency","MntWines","MntFruits","MntMeatProducts"]
for c in cols:
    mn = df[c].min()
    mx = df[c].max()
    df[c] = (df[c] - mn) / (mx - mn)

print(JC)
print(SMC)
print(cosine(vec1,vec2))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(JC, annot=True, cmap="viridis", ax=axes[0])
axes[0].set_title("Jaccard Coefficient")
sns.heatmap(SMC, annot=True, cmap="viridis", ax=axes[1])
axes[1].set_title("Simple Matching Coefficient")
sns.heatmap(COS, annot=True, cmap="viridis", ax=axes[2])
axes[2].set_title("Cosine Similarity")
plt.tight_layout()
plt.show()
