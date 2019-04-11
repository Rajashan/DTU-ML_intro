 # -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:36:35 2017

@author: rajaz
"""

# exercise 2.1.1
import os
import numpy as np
import xlrd

# Load xls sheet with data
os.chdir("C:\\Users\\rajaz\\OneDrive\\Documents\\Machine\\project_data\\")
doc = xlrd.open_workbook(".\\glass_data.xls").sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 1, 10)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(10, 1, 214)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(6)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((213, 9)))
for i, col_id in enumerate(range(1, 10)):
    X[:, i] = np.mat(doc.col_values(col_id, 1, 214)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# exercise 2.1.2

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
plot(X[:, i], X[:, j], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
f.hold()
title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y.ravel()==c
    plot(X[class_mask,i], X[class_mask,j], 'o')

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

# exercise 2.1.3

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
show()

# exercise 2.1.4


from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V = V.T
# Project the centered data onto principal component space
Z = Y * V


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
f.hold()
title('NanoNose data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y.ravel()==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


from scipy.linalg import svd

# (requires data structures from ex. 2.2.1 and 2.2.3)
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V=V.T


print(V[:,1].T)
## Projection of water class onto the 2nd principal component.
# Note Y is a numpy matrix, while V is a numpy array. 

# Either convert V to a numpy.mat and use * (matrix multiplication)
print((Y[y.ravel()==4,:] * np.mat(V[:,1]).T).T)

# Or interpret Y as a numpy.array and use @ (matrix multiplication for np.array)
#print( (np.asarray(Y[y.A.ravel()==4,:]) @ V[:,1]).T )

from matplotlib.pyplot import figure, subplot, hist, hold, xlabel, ylim, show, xticks, boxplot, yticks
import numpy as np

figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()


figure(figsize=(12,10))
hold(True)
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()


boxplot(X)
xticks(range(9),attributeNames)
ylabel('')
title('Glass Data - boxplot')
show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,0]
measurements0 = np.asarray(L)
measurements0 = np.squeeze(measurements0)
stats.probplot(measurements0, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,1]
measurements1 = np.asarray(L)
measurements1 = np.squeeze(measurements1)
stats.probplot(measurements1, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,2]
measurements2 = np.asarray(L)
measurements2 = np.squeeze(measurements2)
stats.probplot(measurements2, dist="norm", plot=pylab)
pylab.show()


import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,3]
measurements3 = np.asarray(L)
measurements3 = np.squeeze(measurements3)
stats.probplot(measurements3, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,4]
measurements4 = np.asarray(L)
measurements4 = np.squeeze(measurements4)
stats.probplot(measurements4, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,5]
measurements5 = np.asarray(L)
measurements5 = np.squeeze(measurements5)
stats.probplot(measurements5, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,6]
measurements6 = np.asarray(L)
measurements6 = np.squeeze(measurements6)
stats.probplot(measurements6, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,7]
measurements7 = np.asarray(L)
measurements7 = np.squeeze(measurements7)
stats.probplot(measurements7, dist="norm", plot=pylab)
pylab.show()

import numpy as np 
import pylab 
import scipy.stats as stats
L = X[:,8]
measurements8 = np.asarray(L)
measurements8 = np.squeeze(measurements8)
stats.probplot(measurements8, dist="norm", plot=pylab)
pylab.show()

# VIRKER!!!!
figure(figsize=(14,10))
for c in range(C):
    subplot(2,C/2,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    #class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(X[class_mask,:])
    title('Class: {0}'.format(classNames[c]))
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)

show()
boxplot(X[class_mask,:])

from matplotlib.pyplot import (figure, hold, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show,imshow,colorbar)
                               
figure(figsize=(12,10))
hold(True)
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()


from matplotlib.pyplot import (figure, show, hold)
from mpl_toolkits.mplot3d import Axes3D


# Indices of the variables to plot

figure()
hold('True')
ind = [0, 1, 2]

colors = ['blue', 'green', 'red']

ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X[class_mask,ind[0]], X[class_mask,ind[1]], X[class_mask,ind[2]])

ax.view_init(30, 220)
ax.set_xlabel(attributeNames[ind[0]])
ax.set_ylabel(attributeNames[ind[1]])
ax.set_zlabel(attributeNames[ind[2]])
show()


from scipy.stats import zscore

X_standardized = zscore(X, ddof=1)

figure()
imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(9), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title('Fisher\'s Iris data matrix')
colorbar()

show()

# standard
figure(figsize=(12,6))
title('Wine: Boxplot (standarized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1,M+1), attributeNames, rotation=45)


#Some attributes, maybe if we can see some correlation. 


Attributes = [1,4,5,6]
NumAtr = len(Attributes)

figure(figsize=(12,12))
hold(True)

for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
show()



