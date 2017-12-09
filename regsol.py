import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor



def mytraining(X,Y):
	reg = KernelRidge(kernel = 'rbf', gamma = 0.15, alpha = 0.001)
	reg.fit(X,Y) 
    
	return reg


    
   
def mytrainingaux(X,Y):
	score = [0] * 10
	reg = [0] * 10

	for degree in range(0,10):
		reg[degree] = make_pipeline(PolynomialFeatures(degree+1), Ridge(alpha = 1.1))
		reg[degree].fit(X, Y)
		score[degree] = -cross_val_score( reg[degree], X, Y, cv = 5, scoring = 'neg_mean_squared_error').mean()

	print(score)

	for i in range(0, 10):
		if score[i] == min(score):
			return reg[i]





def myprediction(X,reg):
    Ypred = reg.predict(X)

    return Ypred






# def mytraining(X,Y):
# 	score = [0] * 10
# 	reg = [0] * 10

# 	for deg in range(0,10):
# 		reg[deg] = SVR(degree = deg + 1, kernel = "poly") # isto ta mal, esta linha nao funciona ! preciso de uma regression com 'poly'
# 		reg[deg].fit(X,Y)
# 		score[deg] = -cross_val_score( reg[deg], X, Y, cv = 5, scoring = 'neg_mean_squared_error').mean()

# 	for i in range(0, 10):
# 		if score[i] == min(score):
# 			return reg[i]
