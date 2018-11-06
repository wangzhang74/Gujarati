#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss
import seaborn as sns

file_path = r'D:\Gujarati excel\Table 2_7.xls'
raw_data = pd.read_excel(file_path,header=3,sheet_name='Table 2_7')
fig = plt.figure()
axis1 = fig.add_subplot(1,2,1)
axis1.scatter(raw_data['M Labor Part.'],raw_data['M Civil Unemp'])
axis2 = fig.add_subplot(1,2,2)
axis2.scatter(raw_data['F Labor Part'],raw_data['F Civil Unemp'])
x_man = raw_data['M Labor Part.']
y_man = raw_data['M Civil Unemp']
#y_man = beta0 + beta1*x_man+u
beta1 = np.sum((x_man-np.mean(x_man))*(y_man-np.mean(y_man)))/np.sum((x_man-np.mean(x_man))**2)
beta0 = np.mean(y_man)-beta1*np.mean(x_man)
y_man_estimate = beta0 + beta1*x_man
R_squared = np.sum((y_man_estimate-np.mean(y_man))**2)/np.sum((y_man-np.mean(y_man))**2)
sigam_squared_estimate = np.sum((y_man-y_man_estimate)**2)/(y_man.count()-2)
t = beta1*math.sqrt(np.sum((x_man-np.mean(x_man))**2))/math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_man.count()-2))  # t分位值
r_xy = np.sum((x_man-np.mean(x_man))*(y_man-np.mean(y_man)))/math.sqrt(np.sum((x_man-np.mean(x_man))**2)*np.sum((y_man-np.mean(y_man))**2))
axis1.plot(x_man,y_man_estimate)
print('beta1='+str(beta1),'beta0='+str(beta0),'R**2='+str(R_squared),'s**2='+str(sigam_squared_estimate)
      ,'t='+str(t),'t_score='+str(t_score),'Passing the test of significance:'+str(abs(t) > t_score)
      ,'r_xy='+str(r_xy))

x_fman = raw_data['F Labor Part']
y_fman = raw_data['F Civil Unemp']
#y_fman = beta_0 + beta_1*x_fman
beta_1 = np.sum((x_fman-np.mean(x_fman))*(y_fman-np.mean(y_fman)))/np.sum((x_fman-np.mean(x_fman))**2)
beta_0 = np.mean(y_fman)-beta_1*np.mean(x_fman)
y_fman_estimate = beta_0 + beta_1*x_fman
R_squared_1 = np.sum((y_fman_estimate-np.mean(y_fman))**2)/np.sum((y_fman-np.mean(y_fman))**2)
sigam_squared_estimate_1 = np.sum((y_fman-y_fman_estimate)**2)/(y_fman.count()-2)
t_1 = beta_1*math.sqrt(np.sum((x_fman-np.mean(x_fman))**2))/math.sqrt(sigam_squared_estimate)
t_score_1 = ss.t.isf(0.025, df=(y_fman.count()-2))  # t分位值
r_xy_1 = np.sum((x_fman-np.mean(x_fman))*(y_fman-np.mean(y_fman)))/math.sqrt(np.sum((x_fman-np.mean(x_fman))**2)*np.sum((y_fman-np.mean(y_fman))**2))
axis2.plot(x_fman,y_fman_estimate)
print('beta1='+str(beta_1),'beta0='+str(beta_0),'R**2='+str(R_squared_1),'s**2='+str(sigam_squared_estimate_1)
      ,'t='+str(t_1),'t_score='+str(t_score_1),'Passing the test of significance:'+str(abs(t_1) > t_score_1)
      ,'r_xy='+str(r_xy_1))

