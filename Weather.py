from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')
data = pd.read_excel(xls, 'Sheet1BC')
#print(data)
#print(len(data))
#print(data.shape)
#January = data["January"]
#print(Shape.January)
Temp = data.iloc[0:12, 16:18]

# print(Temp1)

month = np.arange(1, 13, 1)#print(12 month)
#one = np.ones((month.shape[0], 1))
#newMonth = np.column_stack([one, month])
#print(averageTemp)
#print(month)
#newMonth.insert(loc=0, column='A', value=one) #add 1 in each row of column A

print(Temp)
print(month)
X = month
y = Temp1
plt.plot(X, y, 'ro')
plt.axis([0, 12, 10, 35])
plt.xlabel('Month ')
plt.ylabel('Temp ( C)')
plt.show()
