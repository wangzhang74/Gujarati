#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss
import re

file_path = r'D:\Gujarati excel\Table 5_9.xls'
data_cook = pd.read_excel(file_path, header=3, sheet_name='Table 5_9')
data_cook=data_cook.dropna(axis=0) #去掉空值行

def convert_currency(value):
    '''
    -移除空格、，号、$号
    -生成pure的数值
    '''
    new_value = value.replace('$',' ').replace(',','')#清除$的符号和，号
    index_pattern = re.compile(r'\s+')#compile是把正则表达编译
    lt_clean = re.split(index_pattern,new_value)#maxsplit是按照pattern分割的次数不打这个参数就是默认不限次数
    return np.float(lt_clean.pop())#把str转换为float
''' for i in lt:
        try:
            if type(float(i)) is float:
                new_value = float(i)
                return np.float(new_value)#return是会直接结束所有循环退出def的，后面的return不会执行
            else:
                pass
        except ValueError:
            pass'''


'''for i in target_lt:
    try:
        if type(float(i)) is float:
            new_value = float(i)
            new_lt.append(new_value)
        else:
            pass
    except ValueError:
            pass
'''
price_defined_by_local_currency = data_cook['BMACLC'].apply(convert_currency)
#e_real = data_cook['Actual $ exchange rate Jan 31st, 2006']*data_cook['BMAC$']/price_defined_by_local_currency  实际利率等于本币名义利率*（本国商品价格/外国商品价格）
e_name = data_cook['Actual $ exchange rate Jan 31st, 2006']*price_defined_by_local_currency/data_cook['BMAC$']
print(map(lambda x:'%.20f'%x,list(e_name)))
#e_real = beta0 + beta1*PPP+u  ppp(Purchasing power parity)即购买力平价，等于当地价格除以美国价格

ppp = data_cook['Implied PPP* of the dollar']
e_real = data_cook['Actual $ exchange rate Jan 31st, 2006']

beta1 = np.sum((ppp - np.mean(ppp)) * (e_real - np.mean(e_real))) / np.sum((ppp - np.mean(ppp)) ** 2)
beta0 = np.mean(e_real) - beta1 * np.mean(ppp)
e_real_estimate = beta0 + beta1 * ppp
residual = e_real - e_real_estimate
R_squared = np.sum((e_real_estimate - np.mean(e_real)) ** 2) / np.sum((e_real - np.mean(e_real)) ** 2)
sigam_squared_estimate = np.sum((e_real - e_real_estimate) ** 2) / (e_real.count() - 2)
standard_residual = residual/math.sqrt(sigam_squared_estimate)
s_squared_beta1 = sigam_squared_estimate/np.sum((ppp - np.mean(ppp)) ** 2)
s_squared_beta0 = sigam_squared_estimate * np.sum(ppp * ppp) / (e_real.count() * np.sum((ppp - np.mean(ppp)) ** 2))
t_value = beta1 * math.sqrt(np.sum((ppp - np.mean(ppp)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(e_real.count() - 2))  # t分位值
p = ss.t.sf(t_value, e_real.count() - 2)  #p值
r_xy = np.sum((ppp - np.mean(ppp)) * (e_real - np.mean(e_real))) / math.sqrt(np.sum((ppp - np.mean(ppp)) ** 2) * np.sum((e_real - np.mean(e_real)) ** 2))

print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1)),'p='+'%.15f'%p     #格式化字符串%s，格式化整数%d，格式化浮点数%f前面的示例确定了小数点后的精度
      ,'t=' + str(t_value), 't_score=' + str(t_score), 'Passing the test of significance:' + str(abs(t_value) > t_score)
      ,'r_xy=' + str(r_xy),'confidence interval of beta1: [' + str(beta1-t_score*math.sqrt(s_squared_beta1)) +' , ' + str(beta1+t_score*math.sqrt(s_squared_beta1)) +']'
      ,'confidence interval of beta0: [' + str(beta0 - t_score * math.sqrt(s_squared_beta0)) + ' , ' + str(beta0 + t_score * math.sqrt(s_squared_beta0)) + ']')

fig = plt.figure()
axis1 = fig.add_subplot(2,1,1)
axis1.scatter(ppp,e_real,color='r',alpha=0.5,marker='o')
axis1.set_title('Real Exchange Rate Related To PPP')
axis1.set_xlabel('PPP')
axis1.set_ylabel('Real Exchange Rate')
axis1.plot(ppp,e_real_estimate)

ln_ppp = ppp.apply(math.log)
ln_e_real = e_real.apply(math.log)

beta1 = np.sum((ln_ppp - np.mean(ln_ppp)) * (ln_e_real - np.mean(ln_e_real))) / np.sum((ln_ppp - np.mean(ln_ppp)) ** 2)
beta0 = np.mean(ln_e_real) - beta1 * np.mean(ln_ppp)
ln_e_real_estimate = beta0 + beta1 * ln_ppp
residual = ln_e_real - ln_e_real_estimate
R_squared = np.sum((ln_e_real_estimate - np.mean(ln_e_real)) ** 2) / np.sum((ln_e_real - np.mean(ln_e_real)) ** 2)
sigam_squared_estimate = np.sum((ln_e_real - ln_e_real_estimate) ** 2) / (ln_e_real.count() - 2)
standard_residual = residual/math.sqrt(sigam_squared_estimate)
s_squared_beta1 = sigam_squared_estimate/np.sum((ln_ppp - np.mean(ln_ppp)) ** 2)
s_squared_beta0 = sigam_squared_estimate * np.sum(ln_ppp * ln_ppp) / (ln_e_real.count() * np.sum((ln_ppp - np.mean(ln_ppp)) ** 2))
t_value = beta1 * math.sqrt(np.sum((ln_ppp - np.mean(ln_ppp)) ** 2)) / math.sqrt(sigam_squared_estimate)
t_score = ss.t.isf(0.025, df=(ln_e_real.count() - 2))  # t分位值
p = ss.t.sf(t_value, ln_e_real.count() - 2)  #p值
r_xy = np.sum((ln_ppp - np.mean(ln_ppp)) * (ln_e_real - np.mean(ln_e_real))) / math.sqrt(np.sum((ln_ppp - np.mean(ln_ppp)) ** 2) * np.sum((ln_e_real - np.mean(ln_e_real)) ** 2))

print('beta1=' + str(beta1), 'beta0=' + str(beta0), 'R**2=' + str(R_squared), 's**2=' + str(sigam_squared_estimate)
      ,'s_beta0=' + str(math.sqrt(s_squared_beta0)),'s_beta1=' + str(math.sqrt(s_squared_beta1)),'p='+'%.15f'%p     #格式化字符串%s，格式化整数%d，格式化浮点数%f前面的示例确定了小数点后的精度
      ,'t=' + str(t_value), 't_score=' + str(t_score), 'Passing the test of significance:' + str(abs(t_value) > t_score)
      ,'r_xy=' + str(r_xy),'confidence interval of beta1: [' + str(beta1-t_score*math.sqrt(s_squared_beta1)) +' , ' + str(beta1+t_score*math.sqrt(s_squared_beta1)) +']'
      ,'confidence interval of beta0: [' + str(beta0 - t_score * math.sqrt(s_squared_beta0)) + ' , ' + str(beta0 + t_score * math.sqrt(s_squared_beta0)) + ']')

axis2 = fig.add_subplot(2,1,2)
axis2.scatter(map(math.log,ppp),map(math.log,e_real),color='m',alpha=0.5,marker='o')
axis2.plot(ln_ppp,ln_e_real_estimate)
axis2.set_xlabel('LnPPP')
axis2.set_ylabel('Ln(Real Exchange Rate)')
axis2.set_title('Real Exchange Rate Related To PPP')