#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from typing import Any, Union

file_path = r'D:\Gujarati excel\Table 2_8.xls'
raw_data = pd.read_excel(file_path,header=3,sheet_name='Table 2_8')
fig = plt.figure()
axis = fig.add_subplot(1,1,1)
axis.scatter(raw_data['TOTALEXP'],raw_data['FOODEXP'])
x_food_expend = raw_data['TOTALEXP']
y_totall_income = raw_data['FOODEXP']
#y_totall_income = beta0 + beta1*x_food_expend+u
beta1 = np.sum((x_food_expend - np.mean(x_food_expend)) * (y_totall_income - np.mean(y_totall_income))) / np.sum((x_food_expend - np.mean(x_food_expend)) ** 2)
beta0 = np.mean(y_totall_income) - beta1 * np.mean(x_food_expend)
y_totall_estimate = beta0 + beta1 * x_food_expend
R_squared = np.sum((y_totall_estimate - np.mean(y_totall_income)) ** 2) / np.sum((y_totall_income - np.mean(y_totall_income)) ** 2)
sigam_squared_estimate = np.sum((y_totall_income - y_totall_estimate) ** 2) / (y_totall_income.count() - 2)
t = beta1 * math.sqrt(np.sum((x_food_expend - np.mean(x_food_expend)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_totall_income.count() - 2))  # t分位值
r_xy = np.sum((x_food_expend - np.mean(x_food_expend)) * (y_totall_income - np.mean(y_totall_income))) / math.sqrt(np.sum((x_food_expend - np.mean(x_food_expend)) ** 2) * np.sum((y_totall_income - np.mean(y_totall_income)) ** 2))
axis.plot(x_food_expend, y_totall_estimate)
axis.set_xlabel('Totall Income')
axis.set_ylabel('Food Expend')
print('beta1='+str(beta1),'beta0='+str(beta0),'R**2='+str(R_squared),'s**2='+str(sigam_squared_estimate)
      ,'t='+str(t),'t_score='+str(t_score),'Passing the test of significance:'+str(abs(t) > t_score)
      ,'r_xy='+str(r_xy))
#很显然，绘出的图存在异方差现象，而且是递增型的异方差，即u的方差是和x成比例
#那么需要修正原来的回归方程y_totall_income/sqrt(x_food_expend) = beta0/sqrt(x_food_expend) + beta1*sqrt(x_food_expend)+u/sqrt(x_food_expend)