# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 01:38:49 2017

@author: rajaz
"""
import numpy as np
from matplotlib.pylab import (figure, bar,hist,plot, semilogx, imshow, xticks, yticks, loglog, xlabel, ylabel, legend, title, clim, subplot, show, hold, plot, contour, contourf, cm, colorbar, legend,ylim)
import os
import xlrd
from sklearn.mixture import GaussianMixture
from sklearn import cross_validation
from scipy.stats.kde import gaussian_kde
from sklearn.neighbors import NearestNeighbors

os.chdir("C:\\Users\\rajaz\\OneDrive\\Documents\\Machine\\project_data\\")
doc = xlrd.open_workbook(".\\glass_data.xls").sheet_by_index(0)

attributeNames = doc.row_values(0, 1, 10)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(10, 1, 214)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(6)))

y = np.array([classDict[value] for value in classLabels])
y = y.reshape(213,)
y = y.astype(float)

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((213, 9)))
for i, col_id in enumerate(range(1, 10)):
    X[:, i] = np.mat(doc.col_values(col_id, 1, 214)).T
X = np.asarray(X)
X = X-X.mean(axis=0)
# Compute values of N (number of observations), M (number of attributes) and C (classes).
N = len(y)
M = len(attributeNames)
C = len(classNames)
C = C


#%%


from scipy.linalg import svd

U, s, V = svd(X)
k1 = V.T[:, 0]
k2 = V.T[:, 1]

W2 = V.T[:, :2]
X2D = np.dot(X,W2)

Q = X2D
i = 0
j = 1

f = figure()
f.hold()
title('Glass Data PCA')
#Z = array(Z)
for p in range(C):
    # select indices belonging to class c:
    class_mask = y.ravel()==p
    plot(Q[class_mask,i], Q[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))


#%%

from toolbox_02450 import clusterplot, clusterval
from sklearn.cluster import k_means

# Number of clusters:
K = 5

# K-means clustering:
centroids, cls, inertia = k_means(X,K)
    
# Plot results:
figure(figsize=(14,9))
clusterplot(X, cls, centroids, y)
show()


# Maximum number of clusters:
K = 10

# Allocate variables:
Rand = np.zeros((K,))
Jaccard = np.zeros((K,))
NMI = np.zeros((K,))


for k in range(K):
    # run K-means clustering:
    #cls = Pycluster.kcluster(X,k+1)[0]
    centroids, cls, inertia = k_means(X,k+1)
    # compute cluster validities:
    Rand[k], Jaccard[k], NMI[k] = clusterval(y,cls)    
        
# Plot results:

figure(1)
title('Cluster validity')
plot(np.arange(K)+1, Rand)
plot(np.arange(K)+1, Jaccard)
plot(np.arange(K)+1, NMI)
ylim(-2,1.1)
legend(['Rand', 'Jaccard', 'NMI'], loc=4)
show()

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


Method = 'single'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()






#%% Day 11

# Number of clusters
K = 10
cov_type = 'diag'       
# type of covariance, you can try out 'diag' as well
reps = 1                
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type == 'diag':    
    new_covs = np.zeros([K,M,M])    

count = 0    
for elem in covs:        
    temp_m = np.zeros([M,M])        
    for i in range(len(elem)):            
        temp_m[i][i] = elem[i]        
    
    new_covs[count] = temp_m        
    count += 1
        
covs = new_covs
# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()



# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = cross_validation.KFold(N,10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}\n'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV:

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results

figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()

x = np.linspace(-10, 10, 50)
# Compute kernel density estimate
kde = gaussian_kde(X.ravel())
xe = np.linspace(-10, 10, 100)
# Plot kernel density estimate
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, kde.evaluate(xe))
title('Kernel density estimate')
show()






