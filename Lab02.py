import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# A1

data = pd.read_excel('Lab_Session_Data.xlsx',sheet_name=0)

X = data.iloc[:,1:4].values
y = data.iloc[:,4].values

def cal_matrix_rank(X):
    return np.linalg.matrix_rank(X)
print(f"the rank of the matrix is : {cal_matrix_rank(X)}")

def cal_weights(x,y):
    return np.linalg.pinv(x).dot(y)
print(f"the weight matrix from the data is : {cal_weights(X,y)}")

def classifier(y):
    result = []
    for num in y:
        if num>200:
            result.append("RICH")
        else:
            result.append("POOR")
    return result

print(f"the classification of the standards (rich or poor) based on the data is : {classifier(y)}")

#A3
# the calculation of the mean and the variance of the price data
 
stock_data = pd.read_excel('Lab_Session_Data.xlsx',sheet_name=1)
price_data = stock_data.iloc[:,3].values
def cal_mean_var_price(price):
    mean = np.mean(price)
    var = np.var(price)
    return mean,var
mean_price,var_price = cal_mean_var_price(price_data)
print(f"mean of price data : {mean_price} and the variance : {var_price}")

# the manual calculationd of the meana nd the variance and checking the comparision results between the manual and the buitl in calculations of the mean and the variance
def cal_mean_var(price):
    mean = sum(list(price))/len(price)
    var = sum((list(price)-mean)**2)
    return mean,var
print(f"calculated mean : {cal_mean_var(price_data)} and variance : {cal_mean_var(price_data)}")

# the calculation of the mean of the wednesday prices and comparing the results with the population mean

wednesday_prices = stock_data[stock_data["Day"]=="Wed"].iloc[:,3].values
def wed_mean(price):
    return np.mean(price)
wed_mean_value = wed_mean(wednesday_prices)
print(f"mean of the wednesday prices : {wed_mean_value}")

# the calculation of the mean of the price data of the month april and comparing the results with the population mean

price_data_april = stock_data[stock_data["Month"]=="Apr"].iloc[:,3].values
def april_mean(price):
    return np.mean(price)
april_mean_value = april_mean(price_data_april)
print(f"the mean of the april month prices : {april_mean_value}")

# the probability of making the loss over the stock
loss_fn = lambda x : x<0
loss = loss_fn(stock_data.iloc[:,8].values)
probability = len(loss[loss == True])/len(stock_data.iloc[:,8])
print(f"the probability of the loss over the stock : {probability}")

# probability of getting the profit on wednesday

profit_fn = lambda x: x>0
profit = profit_fn(stock_data[stock_data["Day"]=="Wed"].iloc[:,8].values)
probability_profit = len(profit[profit == True])/len(stock_data.iloc[:,8])
print(f"the probability of getting the profit on wednesdays : {probability_profit}")

# probability of making profit given today is wednesday

probability_it_is_wed = len(stock_data[stock_data["Day"]=="Wed"])/len(stock_data.iloc[:,1])
print(f"the probability is : {probability_profit/probability_it_is_wed}")

# the scatter plot of the chg% data against the day of the week
chg_data = stock_data.iloc[:,8].values
day_of_week_data = stock_data.iloc[:,2].values
def scatter_plot(x,y):
    plt.scatter(chg_data,day_of_week_data)
    plt.xlabel("chg%")
    plt.ylabel("day of the week")
    plt.title("chg% vs day of the week")
    plt.show()
scatter_plot(chg_data,day_of_week_data)
# A4

thyroid0387_UCI_data = pd.read_excel('Lab_Session_Data.xlsx',sheet_name=2)

numerical_data = thyroid0387_UCI_data.iloc[:,[1,18,20,22,24,26,28]].values

only_numeric_fn = lambda x : x[x!="?"]
def cal_mean_numeric(num_data):
    mean = []
    for i in range(num_data.shape[1]):
        only_numeric = only_numeric_fn(num_data[:,i])
        mean.append(float(np.mean(only_numeric)))
    return mean
mean_numerical = cal_mean_numeric(numerical_data)
print(f" the mean of the numerical data columns are : {mean_numerical}")

# A5

marketing_campaign = pd.read_excel('Lab_Session_Data.xlsx',sheet_name=3)
binary_data = marketing_campaign.iloc[[0,1],[5,20,21,22,23,24,25,28]]

# calculation of the Jaccob Coefficient, SMC

def cal_f(data):
    f00 = 0
    f01 = 0
    f10 = 0
    f11 = 0
    for i in range(binary_data.shape[1]):
        if binary_data.iloc[0,i] and binary_data.iloc[1,i] == 0:
            f00+=1
        elif binary_data.iloc[0,i]==0 and binary_data.iloc[1,i]==1:
            f01+=1
        elif binary_data.iloc[0,i]==1 and binary_data.iloc[1,i]==0:
            f10+=1
        else:
            f11+=1
    return f00,f01,f10,f11

f00,f01,f10,f11 = cal_f(binary_data)
def cal_jc_smc(f00,f01,f10,f11):
    jc = f11/(f01+f10+f11)
    smc = (f00+f11)/(f00+f01+f10+f11)
    return jc,smc
jc,smc = cal_jc_smc(f00,f01,f10,f11)
print(f"jc : {jc} and smc : {smc}")

# A6
#cosine similarity measure

data_for_cosine = marketing_campaign.iloc[:2,[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]].values
def conv_to_num(data):
    for i in range(data.shape[0]):
        if data[i,1]=="Graduation" and data[i,2]=="Single":
            data[i,1] = 1
            data[i,2] = 0
    return data
numeric_data_cosine = conv_to_num(data_for_cosine)

def cal_cosine_similarity(data):
    return np.dot(data[0],data[1])/len(data[0])*len(data[1])

cosine_similarity = cal_cosine_similarity(numeric_data_cosine)
print(f"the cosine similarity between these two document vectors is : {cosine_similarity}")

# a7
all_attributes_data = marketing_campaign.iloc[:20,:].values
print(all_attributes_data) 