Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 22:20:52) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> rom __future__ import division, print_function, unicode_literals
SyntaxError: invalid syntax
>>> from __future__ import division, print_function, unicode_literals
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')

>>>data = pd.read_excel(xls, 'Sheet1BC')
SyntaxError: multiple statements found while compiling a single statement
>>> xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')

>>>data = pd.read_excel(xls, 'Sheet1BC')
SyntaxError: multiple statements found while compiling a single statement
>>> xls = pd.ExcelFile('D:\AHocMay\BTL\VN.csv')
>>> data = pd.read_excel(xls, 'Sheet1BC')
>>> for i in range(0, 12):
	y = np.array( data.iloc[16*i: 16*i+8 , 1:2])
	y = np.array(y)	
	X = np.asmatrix( np.arange(2002, 2010))
	X = np.array(X)	
	one = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((one, X), axis = 1)
	A = np.dot(Xbar.T, Xbar)
	b = np.dot(Xbar.T, y)
	w = np.dot(np.linalg.pinv(A), b)
	w_0 = w[0][0]
	w_1 = w[1][0]
	dd = []
	y = []
	for i in range(2010, 2018):
		dd.append(w_0 + w_1*i)
		dd = np.asmatrix(L).T
		dd = dd[0:10]
		dd = np.array(dd)
	tt = data.iloc[16*i+8:16*(i+1), 1:2]
	tt = np.asmatrix(tt)
	tt = np.array(tt)

	
Traceback (most recent call last):
  File "<pyshell#10>", line 9, in <module>
    b = np.dot(Xbar.T, y)
  File "<__array_function__ internals>", line 6, in dot
ValueError: shapes (9,1) and (8,1) not aligned: 1 (dim 1) != 8 (dim 0)
>>> for i in range(0, 12):
	y = np.array( data.iloc[16*i: 16*i+8 , 1:2])
	y = np.array(y)
	X = np.asmatrix( np.arange(2002, 2010)).T
	X = np.array(X)
	one = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((one, X), axis = 1)
	A = np.dot(Xbar.T, Xbar)
	b = np.dot(Xbar.T, y)
	w = np.dot(np.linalg.pinv(A), b)
	w_0 = w[0][0]
	w_1 = w[1][0]
	dd = []
	y = []
	for i in range(2010, 2018):
		dd = np.append(dd, w_0 + w_1*i)
		dd = np.asmatrix(dd).T
		dd = dd[0:10]
		dd = np.array(dd)
	tt = data.iloc[16*i+8:16*(i+1), 1:2]
	tt = np.asmatrix(tt)
	tt = np.array(tt)

	
>>> print(dd)
[[17.58214289]
 [17.53095242]
 [17.47976195]
 [17.42857148]
 [17.37738101]
 [17.32619054]
 [17.27500007]
 [17.2238096 ]]
>>> print(tt)
[]
>>> tt = data.iloc[8:16, 1:2]
>>> tt = np.array(tt)
>>> print(tt)
[[15.4]
 [15.1]
 [16.2]
 [17. ]
 [15.8]
 [16.7]
 [15.9]
 [16.7]]
>>> L = []
>>> for i in range(0,8):
	if ( abs( tt[i] - dd[i] ) < 3):
		L = np.append(L, 1)

	else:
		L= np.append(L, 0)

		
>>> print(L)
[1. 1. 1. 1. 1. 1. 1. 1.]
>>> for i in range(0,8):
	if ( abs( tt[i] - dd[i] ) < 1.5):
		L = np.append(L, 1)

	else:
		L= np.append(L, 0)

		
>>> print(L)
[1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 1. 1. 1.]
>>> L=[]
>>> for i in range(0,8):
	if ( abs( tt[i] - dd[i] ) < 1.5):
		L = np.append(L, 1)

	else:
		L= np.append(L, 0)

	
>>> print(L)
[0. 0. 1. 1. 0. 1. 1. 1.]
>>> y = L
>>> X = tt
>>> print(X)
[[15.4]
 [15.1]
 [16.2]
 [17. ]
 [15.8]
 [16.7]
 [15.9]
 [16.7]]
>>> X = X.T
>>> print(X)
[[15.4 15.1 16.2 17.  15.8 16.7 15.9 16.7]]
>>> print(type(X))
<class 'numpy.ndarray'>
>>> X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
>>> def sigmoid(s):
	    return 1/(1 + np.exp(-s))

	
>>> def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

>>> eta = .05
>>> d = X.shape[0]
>>> w_init = np.random.randn(d, 1)
>>> 
>>> w = logistic_sigmoid_regression(X, y, w_init, eta)
>>> print(w[-1])
[[-8.16830697]
 [ 1.09289005]]
>>> print(sigmoid(np.dot(w[-1].T, X)))
[[0.99982703 0.99975993 0.99992784 0.9999699  0.99988827 0.99995822
  0.99989984 0.99995822]]
>>> def draw(X,y):
	X0 = X[1, np.where(y == 0)][0]
	y0 = y[np.where(y == 0)]
	X1 = X[1, np.where(y == 1)][0]
	y1 = y[np.where(y == 1)]

	plt.plot(X0, y0, 'ro', markersize = 8)
	plt.plot(X1, y1, 'bs', markersize = 8)

	xx = np.linspace(0, 6, 1000)
	w0 = w[-1][0][0]
	w1 = w[-1][1][0]
	threshold = -w0/w1
	yy = sigmoid(w0 + w1*xx)
	plt.axis([10, 30, -1, 2])
	plt.plot(xx, yy, 'g-', linewidth = 2)
	plt.plot(threshold, .5, 'y^', markersize = 8)
	plt.xlabel('tt')
	plt.ylabel('Danhgia')
	plt.show()

	
>>> draw(X,y)
>>> def draw(X,y):
	X0 = X[1, np.where(y == 0)][0]
	y0 = y[np.where(y == 0)]
	X1 = X[1, np.where(y == 1)][0]
	y1 = y[np.where(y == 1)]

	plt.plot(X0, y0, 'ro', markersize = 8)
	plt.plot(X1, y1, 'bs', markersize = 8)

	xx = np.linspace(15, 20, 3000)
	w0 = w[-1][0][0]
	w1 = w[-1][1][0]
	threshold = -w0/w1
	yy = sigmoid(w0 + w1*xx)
	plt.axis([14, 18, -1, 2])
	plt.plot(xx, yy, 'g-', linewidth = 2)
	plt.plot(threshold, .5, 'y^', markersize = 8)
	plt.xlabel('tt')
	plt.ylabel('Danhgia')
	plt.show()

	
>>> draw(X,y)
>>> 
