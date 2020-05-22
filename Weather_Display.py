from __future__ import division, print_function, unicode_literals
import sys
print(sys.version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')
data = pd.read_excel(xls, 'Sheet1BC')

#plot diagram avgmonths of year
TempAvg = data.iloc[0:12, 16:18]
month = np.arange(1, 13, 1)#print(12 month)
#print(TempAvg) print(month)
X = month
y = TempAvg
plt.plot(X, y, 'ro')
plt.axis([0, 12, 10, 35])
plt.title('Display Average Temperature of months from 2002-2017')
plt.xlabel('Month ')
plt.ylabel('Temp ( C)')
#plt.show()

#display 12 diagrams of temp of 12 months
def PlotTemp_Month (Time, Temperature):
    X_year = Time
    y_month = Temperature
    plt.plot(X_year, y_month, 'ro')
    plt.axis([2002, 2017, 5, 35])
    plt.title('Display Temperature of month from 2002 to 2017.')
    plt.xlabel('Year ')
    plt.ylabel('Temp (C)')
    plt.show()
#display 12 diagrams continuos
for i in range(2, 13):
    Timee = data.iloc[13*(i-1):(13*(i-1)+15), 15:16]
    Temp_Feb = data.iloc[13*(i-1):(13*(i-1)+15), 1:2]
    PlotTemp_Month(Timee, Temp_Feb)
        
