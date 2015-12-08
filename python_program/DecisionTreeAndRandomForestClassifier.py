import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

with open('train.txt', 'r') as f:
	train = f.read().splitlines()
f.closed

with open('test.txt', 'r') as f:
	test = f.read().splitlines()
f.closed

X_train = []
y_train = []
X_test = []
y_test = []
rows = len(train)
for line in train[: rows]:
	temp_list1 = map(float,line.split()[0:-1])
	X_train.append(temp_list1)
	y_train.append(float(line.split()[-1]))

rows = len(test)
for line in test[: rows]:
	temp_list2 = map(float,line.split()[0:-1])
	X_test.append(temp_list2)
	y_test.append(float(line.split()[-1]))

numtrain = len(X_train)
numtest = len(X_test)
print ("Pre-setting of paramets: ")
print ("total sumple number is %d" % (numtrain+numtest))
print ("training number is %d" % numtrain)
print ("testing number is %d" % numtest)
''' 
DECISION TREE MODEL
'''
depth = [5, 10, 20, 50, 100]
for d in depth:
	model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=d)
	model.fit (X_train, y_train)
	yhat_test = model.predict(X_test)
	score = model.score(X_test, y_test)
	print (">> Decision tree accuracy is %.4f with max depth %d" % (score, d))
'''
RandomForestClassifier
'''
n_candidate = [3, 5, 20, 40, 60, 100]
for n in n_candidate:
	model = RandomForestClassifier(n_estimators=n, max_depth=None)
	model.fit (X_train, y_train)
	yhat_test = model.predict(X_test)
	score = model.score(X_test, y_test)
	print (">> Random forest accuracy is %.4f with tree number = %d" % (score, n))

depth_rf = [5, 10, 20, 50, 100]
print ("Fix tree number as 40")
for d in depth_rf:
	model = RandomForestClassifier(n_estimators=40, max_depth=d)
	model.fit (X_train, y_train)
	yhat_test = model.predict(X_test)
	score = model.score(X_test, y_test)
	print (">> Random forest accuracy is %.4f with max_depth = %d" % (score, d))