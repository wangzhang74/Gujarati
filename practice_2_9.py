#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

file_path = r'D:\Gujarati excel\Table 2_9.xls'
raw_data = pd.read_excel(file_path,header=3,sheet_name='Table 2_9')
fig = plt.figure()
axis1 = fig.add_subplot(1,2,1)
axis1.plot(raw_data['Year'],raw_data['Male_c'],linestyle = '--',color = 'g',marker='o')
axis1.plot(raw_data['Year'],raw_data['Male_m'],linestyle = '--',color = 'k',marker='o')
axis1.legend(loc='best')
axis2 = fig.add_subplot(1,2,2)
axis2.plot(raw_data['Year'],raw_data['Female_c'],linestyle = '--',color = 'g',marker='o')
axis2.plot(raw_data['Year'],raw_data['Female_m'],linestyle = '--',color = 'k',marker='o')
axis2.legend(loc='best')
fig.show()