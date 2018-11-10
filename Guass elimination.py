# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:52:23 2018

@author: Administrator
"""

import numpy as np

r = int(input('Rows of your equations =?'))
c = int(input('Column of your equations =?'))
lt=[]
signal = True
while signal:
    if len(lt) < r*c:
        add_parameter_coefficient = float(input('Please add your coefficent :'))
        lt.append(add_parameter_coefficient)
    else:
        signal = False

matrix = np.asarray(lt).reshape(r,c,order='C')
if r>=c:
    for col_1 in range(c):#对第一列的操作做循环
#操作行，判定第1列元素是否为0，然后均化为1
        for row_1 in range(col_1,r):
            if matrix[row_1,:][col_1] != 0:
                matrix[row_1,:] = np.true_divide(matrix[row_1,:],matrix[row_1,:][col_1])#numpy的除法（/）默认是floor取整除法，真除要用true_divide
            else:
                pass

#换行然后把主元等于1的行放到第一行，等于0的行放到后面
        column_1 = matrix[col_1:r,col_1].flatten()
        index_arr_1 = np.argsort(-column_1)+col_1#argsort()返回的是排序索引，默认是升序
        matrix[col_1:r] = matrix[index_arr_1]

#把除了第一行以外的行的第一个元素变为0
        for row_2 in range(col_1,r):
            if row_2+1 < r:
                if matrix[row_2+1,:][col_1] == 1:
                    matrix[row_2+1,:] = matrix[row_2+1,:]-matrix[col_1,:]
                else:
                    pass
            elif row_2+1 == r:
                pass
                      
    for col_3 in range(1,c):
        for row_6 in range(col_3):
            if matrix[row_6,:][col_3] !=0:
                matrix[row_6,:] = matrix[row_6,:]-(matrix[col_3,:]*matrix[row_6,:][col_3])
            else:
                pass

elif r<c:
    for col_2 in range(r):#对第一列的操作做循环
#操作行，判定第1列元素是否为0，然后均化为1
        for row_3 in range(col_2,r):
            if matrix[row_3,:][col_2] != 0:
                matrix[row_3,:] = np.true_divide(matrix[row_3,:],matrix[row_3,:][col_2])#numpy的除法（/）默认是floor取整除法，真除要用true_divide
            else:
                pass

#换行然后把主元等于1的行放到第一行，等于0的行放到后面
        column_2 = matrix[col_2:r,col_2].flatten()
        index_arr_2 = np.argsort(-column_2)+col_2#argsort()返回的是排序索引，默认是升序
        matrix[col_2:r] = matrix[index_arr_2]

#把除了第一行以外的行的第一个元素变为0
        for row_4 in range(col_2,r):
            if row_4+1 < r:
                if matrix[row_4+1,:][col_2] == 1:
                    matrix[row_4+1,:] = matrix[row_4+1,:]-matrix[col_2,:]
                else:
                    pass
            elif row_4+1 == r:
                pass    
#向上操作
    for col_4 in range(1,r):
        for row_5 in range(col_4):
            if matrix[row_5,:][col_4] !=0:
                matrix[row_5,:] = matrix[row_5,:]-(matrix[col_4,:]*matrix[row_5,:][col_4])
            else:
                pass


    
    