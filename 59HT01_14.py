from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')
data = pd.read_excel(xls, 'Sheet1BC')
TempAvg = data.iloc[0:12, 16:18]
y = np.asmatrix(TempAvg)
month = np.arange(1, 13, 1)
X = np.asmatrix(month).T
def draw_diagram(X, y):
	plt.plot(X, y, 'bo')
	plt.axis([1, 12, 15, 35])
	plt.xlabel('Month')
	plt.ylabel('Temp *C')
	plt.title("Display temperature of month via 16 years")
	plt.show()
	one = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((one, X), axis = 1)
	A = np.dot(Xbar.T, Xbar)
	b = np.dot(Xbar.T, y)
	w = np.dot(np.linalg.pinv(A), b)
	print('w = ', w)
	w_0 = w[0][0]
	w_1 = w[1][0]
	x0 = np.linspace(15,35, 16)
	y0 = w_0 + w_1*x0
	plt.plot(X.T, y.T, 'ro')    
	plt.plot(x0, y0)             
	plt.axis([1, 12, 15, 35])
	plt.xlabel('Month')
	plt.ylabel('Temp *C')
	plt.title(" Predict temperature of each Month")
	plt.show()
draw(X, y)



X = np.arange(2002, 2018, 1)
X = np.asmatrix(X).T
def draw_diagram_M(X, y):
	plt.plot(X, y, 's', color = '#0066FF')
	plt.axis([2002, 2018, 10, 30])
	plt.xlabel('Month')
	plt.ylabel('Temp *C')
	plt.title("Display temperature of month via 15 years")
	plt.show()
	one = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((one, X), axis = 1)
	A = np.dot(Xbar.T, Xbar)
	b = np.dot(Xbar.T, y)
	w = np.dot(np.linalg.pinv(A), b)
	w_0 = w[0][0]
	w_1 = w[1][0]
	x0 = np.linspace(2002, 2018, 16)
	y0 = w_0 + w_1*x0
	plt.plot(X.T, y.T, 'bo')
	plt.plot(x0, y0)
	plt.axis([2002, 2018, 10, 30])
	plt.xlabel('Month')
	plt.ylabel('Temp *C')
	plt.title(" Predict temperature of each Month")
	plt.show()


for i in range(1, 13, 1):
	y = data.iloc[16*(i-1):16*i, 1:2]
	y = np.array(y)
	draw_diagram_M(X, y)

