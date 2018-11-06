#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

file_path = r'D:\Gujarati excel\Table 5_5.xls'
data_cook = pd.read_excel(file_path, header=4, sheet_name='Table 5_5')
data_cook=data_cook.dropna(axis=0) #去掉空值行
fig = plt.figure()
axis1 = fig.add_subplot(3,1,1)
axis2 = fig.add_subplot(3,1,2)
axis3 = fig.add_subplot(3,2,5)
axis4 = fig.add_subplot(3,2,6)
axis1.scatter(data_cook['SPENDING'],data_cook['SALARY'])
axis1.set_xlabel('Spending');axis1.set_ylabel('Salary')

x_spending = data_cook['SPENDING']
y_salary = data_cook['SALARY']

beta1 = np.sum((x_spending - np.mean(x_spending)) * (y_salary - np.mean(y_salary))) / np.sum((x_spending - np.mean(x_spending)) ** 2)
beta0 = np.mean(y_salary) - beta1 * np.mean(x_spending)
y_salary_estimate = beta0 + beta1 * x_spending
residual = y_salary - y_salary_estimate
R_squared = np.sum((y_salary_estimate - np.mean(y_salary)) ** 2) / np.sum((y_salary - np.mean(y_salary)) ** 2)
sigam_squared_estimate = np.sum((y_salary - y_salary_estimate) ** 2) / (y_salary.count() - 2)
standard_residual = residual/math.sqrt(sigam_squared_estimate)
s_squared_beta1 = sigam_squared_estimate/np.sum((x_spending - np.mean(x_spending)) ** 2)
s_squared_beta0 = sigam_squared_estimate * np.sum(x_spending * x_spending) / (y_salary.count() * np.sum((x_spending - np.mean(x_spending)) ** 2))
t_value = beta1 * math.sqrt(np.sum((x_spending - np.mean(x_spending)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(y_salary.count() - 2))  # t分位值
p = ss.t.sf(t_value,y_salary.count()-2)  #p值 
r_xy = np.sum((x_spending - np.mean(x_spending)) * (y_salary - np.mean(y_salary))) / math.sqrt(np.sum((x_spending - np.mean(x_spending)) ** 2) * np.sum((y_salary - np.mean(y_salary)) ** 2))
axis1.plot(x_spending, y_salary_estimate)
axis1.set_xlabel('SPENDING')
axis1.set_ylabel('SALARY')
axis1.text(6000,20000,'confidence interval of beta1: ['+str(beta1-t_score*math.sqrt(s_squared_beta1))+' , '+str(beta1+t_score*math.sqrt(s_squared_beta1))+']',fontsize=10)
axis1.text(6000,25000,'confidence interval of beta0: [' + str(beta0 - t_score * math.sqrt(s_squared_beta0)) + ' , ' + str(beta0 + t_score * math.sqrt(s_squared_beta0)) + ']',fontsize=10)

print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1)),'p='+'%.15f'%p     #格式化字符串%s，格式化整数%d，格式化浮点数%f前面的示例确定了小数点后的精度
      ,'t=' + str(t_value), 't_score=' + str(t_score), 'Passing the test of significance:' + str(abs(t_value) > t_score)
      ,'r_xy=' + str(r_xy),'confidence interval of beta1: [' + str(beta1-t_score*math.sqrt(s_squared_beta1)) +' , ' + str(beta1+t_score*math.sqrt(s_squared_beta1)) +']'
      ,'confidence interval of beta0: [' + str(beta0 - t_score * math.sqrt(s_squared_beta0)) + ' , ' + str(beta0 + t_score * math.sqrt(s_squared_beta0)) + ']')
spending_new = input('spending=')
estimate_salary = beta0+beta1*spending_new
print('estimate_salary ='+str(estimate_salary))
s_squared_error = sigam_squared_estimate*(1+1/y_salary.count()+(spending_new-np.mean(x_spending))**2/np.sum((x_spending-np.mean(x_spending))**2))
s_error = math.sqrt(s_squared_error)
print('confidence interval of estimate salary: ['+str(estimate_salary-t_score*s_error)+' , '+str(estimate_salary+t_score*s_error)+']')
s_squared_error_mean = sigam_squared_estimate*(1/y_salary.count()+(spending_new-np.mean(x_spending))**2/np.sum((x_spending-np.mean(x_spending))**2))
s_error_mean = math.sqrt(s_squared_error_mean)
print('confidence interval of estimate mean salary: ['+str(estimate_salary-t_score*s_error_mean)+' , '+str(estimate_salary+t_score*s_error_mean)+']')

axis2.hist(standard_residual,histtype='stepfilled',alpha=0.45,orientation='vertical',log=False,label='Frequency of the residual',color='g',density=True,bins=12)#hist是绘制频数直方图,histtype:bar,barstacked,step,stepfilled, bar是通常的图输入dataframe时是紧挨排列柱状图，barstacked是堆砌柱状图输入dataframe时是堆砌柱状图
axis2.plot(standard_residual,ss.norm.pdf(standard_residual,loc=0,scale=1),linestyle='-',marker='o',alpha=0.3,color='r')
#step是柱状图边界线，stepfilled是柱状图边界线且填充,alpha是透明度，log是把x轴的数据转化为对数数据且为0数据会被删除,orientation:horizontal,vertical控制柱状图方向水平或者竖直,label是显示图例legend的时候的各列名称
#color是柱状图颜色，bottom是柱状图底部位移量默认值为0，weights是一组权重数组用来给输入的数据附上权重默认值是各个数据权重为1，density只有true和false默认false当为true时柱状图面积总和为1,bins是直方图的分组的数量

#np.linspace和arange类似，参数有start起始点，stop结束点，num生成的样本点量，endpoint是否包含结束点true和false，和arange的区别就是arange用的步长
x_norm = np.linspace(ss.norm.ppf(0.01),ss.norm.ppf(0.99),100)
axis3.plot(x_norm,ss.norm.cdf(x_norm),alpha=0.4,color='g',label='Normal function',linestyle='--',marker='o')
axis3.set_xlabel('x')
axis3.set_ylabel('standard-norm-cdf')

#The main public methods for continuous RVs are:
#rvs: Random Variates
#pdf: Probability Density Function
#cdf: Cumulative Distribution Function
#sf: Survival Function (1-CDF)
#ppf: Percent Point Function (Inverse of CDF)
#isf: Inverse Survival Function (Inverse of SF)
#stats: Return mean, variance, (Fisher’s) skewness, or (Fisher’s) kurtosis
#moment: non-central moments of the distribution

axis4.plot(residual,ss.norm.cdf(residual,loc=0,scale=math.sqrt(sigam_squared_estimate)),alpha=0.4,color='r',linestyle='--',marker='o')
axis4.set_xlabel('residual')
axis4.set_ylabel('cdf')


#原假设是指随机变量服从正态分布，则JB服从df=2的卡方分布,JB检验是右侧的单边检验
sigam_cap = math.sqrt(np.sum((residual-np.mean(residual))**2)/y_salary.count()) #sigam_cap是有偏估计
skewness = np.sum(((residual-np.mean(residual))/sigam_cap)**3)/y_salary.count()
kurtosis = np.sum(((residual-np.mean(residual))/sigam_cap)**4)/y_salary.count()
JB=y_salary.count()*(skewness**2+(kurtosis-3)**2/4)/6
chi2_score=ss.chi2.isf(0.05,df=2)
P_JB = ss.chi2.sf(JB,df=2)
print('Skewness='+str(skewness),'Kurtosis='+str(kurtosis),'JB='+str(JB),'chi2_score='+str(chi2_score),
      'P_JB=' '%.15f'%P_JB,'passing the examination:'+str(JB<chi2_score))