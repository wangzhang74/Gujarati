#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

file_path = r'D:\Gujarati excel\Table 3_7.xls'
data_cook = pd.read_excel(file_path, header=3, sheet_name='Table 3_7', skipfooter=7)
data_cook=data_cook.dropna(axis=0) #去掉空值行
fig = plt.figure()
axis1 = fig.add_subplot(3,2,1)
axis2 = fig.add_subplot(3,2,2)
axis3 = fig.add_subplot(3,2,3)
axis4 = fig.add_subplot(3,2,4)
axis5 = fig.add_subplot(3, 1, 3)
axis5.plot(data_cook['Year'], data_cook['Gold Price'], linestyle='--', marker='o', color='b')
axis5.plot(data_cook['Year'], data_cook['NYSE'], linestyle='--', marker='o', color='r')
axis5.plot(data_cook['Year'], data_cook['CPI'], linestyle='--', marker='o', color='g')
axis5.legend(loc='best')
axis1.scatter(data_cook['CPI'],data_cook['Gold Price'])
axis2.scatter(data_cook['CPI'],data_cook['NYSE'])

x_CPI = data_cook['CPI']
y_gold_price = data_cook['Gold Price']

beta1 = np.sum((x_CPI - np.mean(x_CPI)) * (y_gold_price - np.mean(y_gold_price))) / np.sum((x_CPI - np.mean(x_CPI)) ** 2)
beta0 = np.mean(y_gold_price) - beta1 * np.mean(x_CPI)
y_gold_price_estimate = beta0 + beta1 * x_CPI
residual = y_gold_price-y_gold_price_estimate
R_squared = np.sum((y_gold_price_estimate - np.mean(y_gold_price)) ** 2) / np.sum((y_gold_price - np.mean(y_gold_price)) ** 2)
sigam_squared_estimate = np.sum((y_gold_price - y_gold_price_estimate) ** 2) / (y_gold_price.count() - 2)
s_squared_beta1 = sigam_squared_estimate/np.sum((x_CPI-np.mean(x_CPI))**2)
s_squared_beta0 = sigam_squared_estimate*np.sum(x_CPI*x_CPI)/(y_gold_price.count()*np.sum((x_CPI-np.mean(x_CPI))**2))
t = beta1 * math.sqrt(np.sum((x_CPI - np.mean(x_CPI)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_gold_price.count() - 2))  # t分位值
r_xy = np.sum((x_CPI - np.mean(x_CPI)) * (y_gold_price - np.mean(y_gold_price))) / math.sqrt(np.sum((x_CPI - np.mean(x_CPI)) ** 2) * np.sum((y_gold_price - np.mean(y_gold_price)) ** 2))
axis1.plot(x_CPI, y_gold_price_estimate)
axis1.set_xlabel('CPI')
axis1.set_ylabel('Gold Price')
print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1))
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))
axis3.plot(residual,ss.norm.cdf(residual,loc=0,scale=math.sqrt(sigam_squared_estimate)),alpha=0.4,color='r',linestyle='-',marker='o')
axis3.set_xlabel('residual')
axis3.set_ylabel('cdf')
#JB_test
sigam_cap = math.sqrt(np.sum((residual-np.mean(residual))**2)/y_gold_price.count()) #sigam_cap是有偏估计
skewness = np.sum(((residual-np.mean(residual))/sigam_cap)**3)/y_gold_price.count()
kurtosis = np.sum(((residual-np.mean(residual))/sigam_cap)**4)/y_gold_price.count()
JB=y_gold_price.count()*(skewness**2+(kurtosis-3)**2/4)/6
chi2_score=ss.chi2.isf(0.05,df=2)
P_JB = ss.chi2.sf(JB,df=2)
print('Skewness='+str(skewness),'Kurtosis='+str(kurtosis),'JB='+str(JB),'chi2_score='+str(chi2_score),
      'P_JB=' '%.15f'%P_JB,'passing the examination:'+str(JB<chi2_score))
axis3.text(100,0.6,'Skewness='+str(skewness),fontsize=8)
axis3.text(100,0.5,'Kurtosis='+str(kurtosis),fontsize=8)
axis3.text(100,0.4,'JB='+str(JB),fontsize=8)
axis3.text(100,0.3,'P_JB=' '%.15f'%P_JB,fontsize=8)


x_CPI = data_cook['CPI']
y_NYSE = data_cook['NYSE']
#y_totall_income = beta0 + beta1*x_food_expend+u
beta1 = np.sum((x_CPI - np.mean(x_CPI)) * (y_NYSE - np.mean(y_NYSE))) / np.sum((x_CPI - np.mean(x_CPI)) ** 2)
beta0 = np.mean(y_NYSE) - beta1 * np.mean(x_CPI)
y_NYSE_estimate = beta0 + beta1 * x_CPI
residual = y_NYSE-y_NYSE_estimate
R_squared = np.sum((y_NYSE_estimate - np.mean(y_NYSE)) ** 2) / np.sum((y_NYSE - np.mean(y_NYSE)) ** 2)
sigam_squared_estimate = np.sum((y_NYSE - y_NYSE_estimate) ** 2) / (y_NYSE.count() - 2)
s_squared_beta1 = sigam_squared_estimate/np.sum((x_CPI-np.mean(x_CPI))**2)
s_squared_beta0 = sigam_squared_estimate*np.sum(x_CPI*x_CPI)/(y_NYSE.count()*np.sum((x_CPI-np.mean(x_CPI))**2))
t = beta1 * math.sqrt(np.sum((x_CPI - np.mean(x_CPI)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_NYSE.count() - 2))  # t分位值
r_xy = np.sum((x_CPI - np.mean(x_CPI)) * (y_NYSE - np.mean(y_NYSE))) / math.sqrt(np.sum((x_CPI - np.mean(x_CPI)) ** 2) * np.sum((y_NYSE - np.mean(y_NYSE)) ** 2))
axis2.plot(x_CPI, y_NYSE_estimate)
axis2.set_xlabel('CPI')
axis2.set_ylabel('NYSE')
print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1))
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))
axis4.plot(residual,ss.norm.cdf(residual,loc=0,scale=math.sqrt(sigam_squared_estimate)),alpha=0.4,color='r',linestyle='-',marker='o')
axis4.set_xlabel('residual')
axis4.set_ylabel('cdf')
#JB_test
sigam_cap = math.sqrt(np.sum((residual-np.mean(residual))**2)/y_NYSE.count()) #sigam_cap是有偏估计
skewness = np.sum(((residual-np.mean(residual))/sigam_cap)**3)/y_NYSE.count()
kurtosis = np.sum(((residual-np.mean(residual))/sigam_cap)**4)/y_NYSE.count()
JB=y_NYSE.count()*(skewness**2+(kurtosis-3)**2/4)/6
chi2_score=ss.chi2.isf(0.05,df=2)
P_JB = ss.chi2.sf(JB,df=2)
print('Skewness='+str(skewness),'Kurtosis='+str(kurtosis),'JB='+str(JB),'chi2_score='+str(chi2_score),
      'P_JB=' '%.15f'%P_JB,'passing the examination:'+str(JB<chi2_score))
axis4.text(500,0.6,'Skewness='+str(skewness),fontsize=8)
axis4.text(500,0.5,'Kurtosis='+str(kurtosis),fontsize=8)
axis4.text(500,0.4,'JB='+str(JB),fontsize=8)
axis4.text(500,0.3,'P_JB=' '%.15f'%P_JB,fontsize=8)