import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def features(X):
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x])					  # numero de letras
        F[x,1] = numVowels(X[x])			  # numero de vogais
        F[x,2] = len(X[x]) - numVowels(X[x])  # numero de consoantes
        F[x,3] = startsWithVowel(X[x])		  # 1 se comeca com vogal, 0 cc
        F[x,4] = sumASCII(X[x])               # soma dos ords da string

    return F     

def mytraining(f,Y):
	n_neigh = 2
	weights = 'distance'
	clf = neighbors.KNeighborsClassifier(n_neigh, weights = weights)
	clf = clf.fit(f,Y)
	return clf


# erro um pouco maior
# def mytraining(f,Y):
# 	min_sample_split = 8
# 	clf = tree.DecisionTreeClassifier(min_samples_split = min_sample_split)
# 	clf = clf.fit(f,Y)
# 	return clf


def myprediction(f, clf):
    Ypred = clf.predict(f)

    return Ypred

def numVowels(str):
	soma = 0

	array = list(map(str.lower().count, "aáàãâeéêiíoóõôuú"))

	for elem in array:
		soma += elem

	return soma

def startsWithVowel(str):
	if len(str.lower()) > 0:
		result = str[0].lower() in "aáàãâeéêiíoóõôuú"
	else:
		result = False

	if result:
		return 1
	else:
		return 0


def sumASCII(string):
	soma = 0
	for elem in string:
		soma += ord(elem)

	return soma