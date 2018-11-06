#encoding utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series

file_path = r'C:\Users\WZ\Downloads\Table 1_4.xls'
raw_date_1 = pd.read_excel(file_path,header=2,sheet_name='Table 1_3')
a=list(raw_date_1['United Kingdom (pound) 2'])
raw_date_1['United Kingdom (pound) 2']=map(lambda x:1/x,a)
country_exchange_rate = list(raw_date_1[:0])
country_exchange_rate.pop(0)
lt_marker=['x','s','o','1','2','3','s','p','8']
zipped_dic = dict(zip(country_exchange_rate,lt_marker))
for k,v in zipped_dic.items():
    plt.scatter(x=raw_date_1['Period'],y=raw_date_1[k],alpha=0.5,marker=v,s=30)
plt.title('Scatter view of exchange rate',fontsize=24)
plt.xlabel('Period',fontsize=14)
plt.ylabel('Exchange rate',fontsize=14)
plt.yscale('log')
plt.legend()
plt.xlim(1985,2006)


