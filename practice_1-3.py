#encoding utf-8
'''read_excel()的参数解释：
io : string, path object ; excel 路径。
sheet_name : string, int, mixed list of strings/ints, or None, default 0 返回多表使用sheetname=[0,1],若sheetname=None是返回全表 注意：int/string 返回的是dataframe，而none和list返回的是dict of dataframe
header : int, list of ints, default 0 指定列名行，默认0，即取第一行，数据为列名行以下的数据 若数据不含列名，则设定 header = None
skiprows : list-like,Rows to skip at the beginning，省略指定行数的数据
skipfooter : int,default 0, 省略从尾部数的int行数据
index_col : int, list of ints, default None指定列为索引列，也可以使用u”strings”
names : array-like, default None, 指定列的名字。
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series

file_path = r'C:\Users\WZ\Downloads\Table 1_3.xls'
raw_data = pd.read_excel(file_path,sheet_name='Table 1_2',header=2) #创建一个主体dataframe
'''
target_column_USA = list(raw_data.USA) #把获取的目标列包装成列表，也可用dataframe_name['column_name']来获取相应列
initial_year_CPI = target_column_USA.pop(0)
inflation_list_USA = []   #美国通胀列表
for x in target_column_USA:
    inflation_rate = (x - initial_year_CPI)/initial_year_CPI   #递推计算通胀率
    inflation_list_USA.append(inflation_rate)  #添加相应通胀率到美国通胀率列表
    initial_year_CPI = x  #初始通胀率重新赋值
raw_data['USA inflation rate'] = 0  #添加一行美国的通胀率到dataframe
insert_USA_ir = Series(inflation_list_USA,index=list(range(1,26))) #把美国的通胀率列表转换成series这个一维数组
raw_data['USA inflation rate'] = insert_USA_ir  #更新dataframe中美国通胀率这一列
'''
#现在考虑重构一个函数.py库，重复生成通胀率的列,见functions
'''这是如何把数据写入excel
   （1）参数excel_writer，输出路径。
   （2）sheet_name，将数据存储在excel的那个sheet页面。
   （3）na_rep，缺失值填充
    (4) colums参数： sequence, optional，Columns to write 选择输出的的列。
    （5）header 参数： boolean or list of string，默认为True,可以用list命名列的名字。header = False 则不输出题头。
    （6）index : boolean, default True Write row names (index) 默认为True，显示index，当index=False 则不显示行索引（名字）。
     index_label : string or sequence, default None 设置索引列的列名。
'''
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax3.plot(np.random.rand(30).cumsum(),'ko--')
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.rand(30))
ax1.hist(np.random.rand(100),bins=20,color='k',alpha=0.)
ax4.plot(x=1,y=2,linestyle='--',color = 'g')
axes = plt.subplots(2,3)
plt.subplots_adjust(wspace=1,hspace=1)
'''这是Series的plot方法的参数使用
data : Series
kind : str

‘line’ : line plot (default)
‘bar’ : vertical bar plot
‘barh’ : horizontal bar plot
‘hist’ : histogram
‘box’ : boxplot
‘kde’ : Kernel Density Estimation plot
‘density’ : same as ‘kde’
‘area’ : area plot
‘pie’ : pie plot
ax : matplotlib axes object

If not passed, uses gca()

figsize : a tuple (width, height) in inches
use_index : boolean, default True

Use index as ticks for x axis

title : string or list

Title to use for the plot. If a string is passed, print the string at the top of the figure. If a list is passed and subplots is True, print each item in the list above the corresponding subplot.

grid : boolean, default None (matlab style default)

Axis grid lines

legend : False/True/’reverse’

Place legend on axis subplots

style : list or dict

matplotlib line style per column

logx : boolean, default False

Use log scaling on x axis

logy : boolean, default False

Use log scaling on y axis

loglog : boolean, default False

Use log scaling on both x and y axes

xticks : sequence

Values to use for the xticks

yticks : sequence

Values to use for the yticks

xlim : 2-tuple/list
ylim : 2-tuple/list
rot : int, default None

Rotation for ticks (xticks for vertical, yticks for horizontal plots)

fontsize : int, default None

Font size for xticks and yticks

colormap : str or matplotlib colormap object, default None

Colormap to select colors from. If string, load colormap with that name from matplotlib.

colorbar : boolean, optional

If True, plot colorbar (only relevant for ‘scatter’ and ‘hexbin’ plots)

position : float

Specify relative alignments for bar plot layout. From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5 (center)

table : boolean, Series or DataFrame, default False

If True, draw a table using the data in the DataFrame and the data will be transposed to meet matplotlib’s default layout. If a Series or DataFrame is passed, use passed data to draw a table.

yerr : DataFrame, Series, array-like, dict and str

See Plotting with Error Bars for detail.

xerr : same types as yerr.
label : label argument to provide to plot
secondary_y : boolean or sequence of ints, default False

If True then y-axis will be on the right

mark_right : boolean, default True

When using a secondary_y axis, automatically mark the column labels with “(right)” in the legend

`**kwds` : keywords

Options to pass to matplotlib plotting method

'''

'''dataframe构造函数
class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)[source]
Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.

Parameters:	
data : numpy ndarray (structured or homogeneous), dict, or DataFrame

Dict can contain Series, arrays, constants, or list-like objects

Changed in version 0.23.0: If data is a dict, argument order is maintained for Python 3.6 and later.

index : Index or array-like

Index to use for resulting frame. Will default to RangeIndex if no indexing information part of input data and no index provided

columns : Index or array-like

Column labels to use for resulting frame. Will default to RangeIndex (0, 1, 2, …, n) if no column labels are provided

dtype : dtype, default None

Data type to force. Only a single dtype is allowed. If None, infer

copy : boolean, default False

Copy data from inputs. Only affects DataFrame / 2d ndarray input'''