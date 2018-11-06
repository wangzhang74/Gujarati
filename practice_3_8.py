#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

file_path = r'D:\Gujarati excel\Table 3_8.xls'
data_cook = pd.read_excel(file_path, header=5, sheet_name='Table 3_8')
data_cook=data_cook.dropna(axis=0) #去掉空值行
fig = plt.figure()
axis1 = fig.add_subplot(2,1,1)
axis2 = fig.add_subplot(2,2,3)
axis3 = fig.add_subplot(2,2,4)
axis1.plot(data_cook['Year'],data_cook['NGDP'],linestyle='--',marker='o',color='k')
axis1.plot(data_cook['Year'],data_cook['RGDP'],linestyle='--',marker='o',color='r')
axis2.scatter(data_cook['Year'],data_cook['NGDP'])
axis3.scatter(data_cook['Year'],data_cook['RGDP'])
axis1.legend(loc='best')
axis1.set_xticks([i for i in range(1959,2006)])#设置刻度的份数
axis1.set_xticklabels([i for i in range(1959,2006)],rotation=30,fontsize='small') #rotation是旋转度，还可以加上一个修改相应刻度值的列表(该列表刻度必须和set_ticks刻度份数相同)

x_year = data_cook['Year']
y_NGDP = data_cook['NGDP']

beta1 = np.sum((x_year - np.mean(x_year)) * (y_NGDP - np.mean(y_NGDP))) / np.sum((x_year - np.mean(x_year)) ** 2)
beta0 = np.mean(y_NGDP) - beta1 * np.mean(x_year)
y_NGDP_estimate = beta0 + beta1 * x_year
R_squared = np.sum((y_NGDP_estimate - np.mean(y_NGDP)) ** 2) / np.sum((y_NGDP - np.mean(y_NGDP)) ** 2)
sigam_squared_estimate = np.sum((y_NGDP - y_NGDP_estimate) ** 2) / (y_NGDP.count() - 2)
s_squared_beta1 = sigam_squared_estimate/np.sum((x_year - np.mean(x_year)) ** 2)
s_squared_beta0 = sigam_squared_estimate * np.sum(x_year * x_year) / (y_NGDP.count() * np.sum((x_year - np.mean(x_year)) ** 2))
t = beta1 * math.sqrt(np.sum((x_year - np.mean(x_year)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_NGDP.count() - 2))  # t分位值
r_xy = np.sum((x_year - np.mean(x_year)) * (y_NGDP - np.mean(y_NGDP))) / math.sqrt(np.sum((x_year - np.mean(x_year)) ** 2) * np.sum((y_NGDP - np.mean(y_NGDP)) ** 2))
axis2.plot(x_year, y_NGDP_estimate)
axis2.set_xlabel('Year')
axis2.set_ylabel('NGDP')
print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1))
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))

x_year = data_cook['Year']
y_RGDP = data_cook['RGDP']

beta1 = np.sum((x_year - np.mean(x_year)) * (y_RGDP - np.mean(y_RGDP))) / np.sum((x_year - np.mean(x_year)) ** 2)
beta0 = np.mean(y_RGDP) - beta1 * np.mean(x_year)
y_RGDP_estimate = beta0 + beta1 * x_year
R_squared = np.sum((y_RGDP_estimate - np.mean(y_RGDP)) ** 2) / np.sum((y_RGDP - np.mean(y_RGDP)) ** 2)
sigam_squared_estimate = np.sum((y_RGDP - y_RGDP_estimate) ** 2) / (y_RGDP.count() - 2)
s_squared_beta1 = sigam_squared_estimate/np.sum((x_year - np.mean(x_year)) ** 2)
s_squared_beta0 = sigam_squared_estimate * np.sum(x_year * x_year) / (y_RGDP.count() * np.sum((x_year - np.mean(x_year)) ** 2))
t = beta1 * math.sqrt(np.sum((x_year - np.mean(x_year)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_RGDP.count() - 2))  # t分位值
r_xy = np.sum((x_year - np.mean(x_year)) * (y_RGDP - np.mean(y_RGDP))) / math.sqrt(np.sum((x_year - np.mean(x_year)) ** 2) * np.sum((y_RGDP - np.mean(y_RGDP)) ** 2))
axis3.plot(x_year, y_RGDP_estimate)
axis3.set_xlabel('Year')
axis3.set_ylabel('RGDP')
print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1))
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))
