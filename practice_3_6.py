#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

file_path = r'D:\Gujarati excel\Table 3_6.xls'
data_cook = pd.read_excel(file_path, header=3, sheet_name='Table 3_6', skipfooter=5)
data_cook=data_cook.dropna(axis=0) #去掉空值行
fig = plt.figure()
axis1 = fig.add_subplot(2,2,1)
axis2 = fig.add_subplot(2,2,2)
axis3 = fig.add_subplot(2,2,3)
axis4 = fig.add_subplot(2,2,4)
axis1.plot(data_cook["Year or quarter"],data_cook['Business sector'],linestyle='--',marker = 'o',color = 'k')
axis1.plot(data_cook["Year or quarter"],data_cook['Nonfarm business sector'],linestyle='--',marker = 'o',color = 'r')
axis1.legend(loc='best')
axis1.set_title('output per hour')
axis2.set_title('wage per hour')
axis2.plot(data_cook["Year or quarter"],data_cook['Business sector.1'],linestyle='--',marker = 'o',color = 'k')
axis2.plot(data_cook["Year or quarter"],data_cook['Nonfarm business sector.1'],linestyle='--',marker = 'o',color = 'r')
axis2.legend(loc='best')
axis3.scatter(data_cook['Business sector'],data_cook['Business sector.1'])
axis4.scatter(data_cook['Nonfarm business sector'],data_cook['Nonfarm business sector.1'])
axis3.set_title('business income determined by output')
axis4.set_title('Non business income determined by output')

x_Non_business_output = data_cook['Business sector']
y_Non_business_income = data_cook['Business sector.1']
#y_totall_income = beta0 + beta1*x_food_expend+u
beta_1 = np.sum((x_Non_business_output - np.mean(x_Non_business_output)) * (y_Non_business_income - np.mean(y_Non_business_income))) / np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2)
beta_0 = np.mean(y_Non_business_income) - beta_1 * np.mean(x_Non_business_output)
y_Non_income_estimate = beta_0 + beta_1 * x_Non_business_output
R_squared = np.sum((y_Non_income_estimate - np.mean(y_Non_business_income)) ** 2) / np.sum((y_Non_business_income - np.mean(y_Non_business_income)) ** 2)
sigam_squared_estimate = np.sum((y_Non_business_income - y_Non_income_estimate) ** 2) / (y_Non_business_income.count() - 2)
t = beta_1 * math.sqrt(np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_Non_business_income.count() - 2))  # t分位值
r_xy = np.sum((x_Non_business_output - np.mean(x_Non_business_output)) * (y_Non_business_income - np.mean(y_Non_business_income))) / math.sqrt(np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2) * np.sum((y_Non_business_income - np.mean(y_Non_business_income)) ** 2))
axis3.plot(x_Non_business_output,y_Non_income_estimate)
axis3.set_xlabel('output')
axis3.set_ylabel('business income')
print('beta1=' + str(beta_1), 'beta0=' + str(beta_0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))

x_Non_business_output = data_cook['Nonfarm business sector']
y_Non_business_income = data_cook['Nonfarm business sector.1']
#y_totall_income = beta0 + beta1*x_food_expend+u
beta_1 = np.sum((x_Non_business_output - np.mean(x_Non_business_output)) * (y_Non_business_income - np.mean(y_Non_business_income))) / np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2)
beta_0 = np.mean(y_Non_business_income) - beta_1 * np.mean(x_Non_business_output)
y_Non_income_estimate = beta_0 + beta_1 * x_Non_business_output
R_squared = np.sum((y_Non_income_estimate - np.mean(y_Non_business_income)) ** 2) / np.sum((y_Non_business_income - np.mean(y_Non_business_income)) ** 2)
sigam_squared_estimate = np.sum((y_Non_business_income - y_Non_income_estimate) ** 2) / (y_Non_business_income.count() - 2)
t = beta_1 * math.sqrt(np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_Non_business_income.count() - 2))  # t分位值
r_xy = np.sum((x_Non_business_output - np.mean(x_Non_business_output)) * (y_Non_business_income - np.mean(y_Non_business_income))) / math.sqrt(np.sum((x_Non_business_output - np.mean(x_Non_business_output)) ** 2) * np.sum((y_Non_business_income - np.mean(y_Non_business_income)) ** 2))
axis4.plot(x_Non_business_output, y_Non_income_estimate)
axis4.set_xlabel('Non output')
axis4.set_ylabel('Non business income')
print('beta1=' + str(beta_1), 'beta0=' + str(beta_0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'t=' + str(t),'t_score=' + str(t_score),'Passing the test of significance:' + str(abs(t) > t_score)
      ,'r_xy=' + str(r_xy))