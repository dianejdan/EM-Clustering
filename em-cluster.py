"""
This is an EM algorithm for clustering
Diane J. Dan, 09/13/2014
"""
    
""" 
Read in data
The data should be in csv format. One record per line.
If you know the label for each record and want to test the performance of clustering,
then the last column in data should be the label.
Otherwise, the data contains only the features. 
Read in initial number of clusters specified by user
"""

import sys
import numpy as np

iris = np.genfromtxt(sys.argv[1], delimiter=',') # I used iris data as a test

k = int(sys.argv[2]) # number of clusters specified by user

""" Initialize model parameters """

if (sys.argv[3] == 'Y') | (sys.argv[3] == 'y'):
    ### since the last column of data is label,
    ### the length of mean vector is column number of data minus one
    dim = iris.shape[1]-1
else:
    dim = iris.shape[1]
	

### means of clusters
### since there are k clusters, there are k mean vectors
mius = np.zeros([dim, k])

### number of data points in initial clusters
### if the total number of data points in iris is not divisible by 
### number of cluster k, then, the first k-1 cluster will have the
### same number [n/k] ([n/k] means the integer part of n/k) of data points,
### the last cluster will have n-[n/k]*(k-1) number of data points.
### the last cluster will have at most (k-1) more data points than the first
### k-1 clusters. This is only one method to resolve the non-divisible problem.
### you can also add one data points in the first k-1 clusters and
### the last cluster will have less data points than the first k-1 clusters,
### at most (k-1) data points less.
numPClus = int(iris.shape[0]/k)

for i in range(k):
    sInd = numPClus*i
    eInd = numPClus*(i+1)
    if ((iris.shape[0]-eInd) < k): ### if the current cluster is the last one
                                   ### it will include all the left data points 
        eInd = iris.shape[0]
    mius[:,i] = np.mean(iris[sInd:eInd, range(dim)], axis=0)
        

### covariance matrices of clusters
### as above explained, the dimension of covariance matrix is
### (column number of iris data minus one) X 
### (column number of iris data minus one)
### there are k covariance matrices
sigmas = np.zeros([k, dim, dim])

### initialize the diagonal entries of covariance matrices to 1
### so covariance matrices are identity matrices
for i in range(k):
    for j in range(dim):
        sigmas[i,j,j] = 1
        

### initialize the probabilities of each clusters
probs = np.zeros([k,1])
probs[:,0] = 1.0/k


""" main iterative EM algorithm """
n = iris.shape[0]  ### sample size in iris data
diff = 1  ### initial difference, should be larger than eps 
eps = 0.001 ### convergence criteria
w = np.zeros([k, n]) ### the weights (probabilities one data point belong to a cluster)

iteration = 0  ### number of iterations 

while diff > eps:
    iteration = iteration+1
    probX = np.zeros([n,1]) ### save sum of probabilities of data points belonging
                            ### to different clusters, total probability

    ### calculate the sum of probabilities of individual data points belonging to
    ### different clusters. The summation in denominator in Step 7 in algorithm 13.3
    for i in range(n):
        for j in range(k):
            ### I don't find multivariate normal pdf function in numpy
            ### so I write my own.
            uSu = np.dot( np.dot(iris[i,range(dim)]-mius[:,j], np.linalg.inv(sigmas[j,:])), \
                             iris[i,range(dim)]-mius[:,j] )/2.0
            mvnPDF = np.exp(-uSu)/np.sqrt(np.power(2*np.pi, dim)*np.linalg.det(sigmas[j,:]))
            probX[i,0] = probX[i,0] + mvnPDF*probs[j,0]
    
    ### calculate posterior probability w_ij = P(cluster_j|x_i), i.e. given probability of data point
    ### calculate the probability it belongs to a cluster. Expectation step in Step 7 in algorithm 13.3
    for i in range(n):
        for j in range(k):
            ### My multivariate normal pdf function
            uSu = np.dot( np.dot(iris[i,range(dim)]-mius[:,j], np.linalg.inv(sigmas[j,:])), \
                             iris[i,range(dim)]-mius[:,j] )/2.0
            mvnPDF = np.exp(-uSu)/np.sqrt(np.power(2*np.pi, dim)*np.linalg.det(sigmas[j,:]))
            w[j,i] = mvnPDF*probs[j,0]/probX[i,0] ### calculate the weight (posterior) w
    
    ### update means, covariance matrices of clusters, the maximization step.
    newMius = np.zeros([dim, k])  ### save new means of clusters
    newSigmas = np.zeros([k, dim, dim])  ### save new covariance matrices of clusters
    newProbs = np.zeros([k,1])  ### save new probabilities of clusters
    for i in range(k):
        newMius[:,i] = np.dot(w[i,:], iris[:,range(dim)])/np.sum(w[i,:]) ### calculate new means
        for j in range(n):
            xTmp = iris[j,range(dim)]-newMius[:,i]
            newSigmas[i,:] = newSigmas[i,:]+w[i,j]*np.outer(xTmp, xTmp)
        newSigmas[i,:] = newSigmas[i,:]/np.sum(w[i,:])  ### calculate new covariance matrices
        newProbs[i,0] = np.sum(w[i,:])/n  ### calculate new probabilities
    ### calculate difference between new means and old means of clusters
    diff = 0
    for i in range(k):
        ### calculate Euclidean distance between new means and old means,
        ### calculate the sum of clusters
        diff = diff+np.dot(newMius[:,i]-mius[:,i],newMius[:,i]-mius[:,i])
    mius = newMius
    sigmas = newSigmas
    probs = newProbs

""" print results """

print 'Mean:'
norms = [0]*k  ### save norms of means of clusters
for i in range(k):
    norms[i] = np.dot(mius[:,i], mius[:,i]) ### calculate norms of means of clusters
inds = np.argsort(norms) ### sort norms with indices
    
for i in range(k):
    print np.round(mius[:,inds[i]]*1000)/1000 ### print means in ascending order of norms
    
print 'Covariance Matrices:'
for i in range(k):
    print np.round(sigmas[inds[i],:]*1000)/1000 ### print covariance matrices in
    print ''   ### ascending order of norms as above

print 'Iteration count=%d' % iteration ### print number of iteration

w = np.zeros([k, n]) ### the weights, P(C_j|x_i), row is cluster, column is data point
probX = np.zeros([n,1]) ### store sum of probabilities of data points belonging
                        ### to different clusters, P(x_i)

### calculate the sum of probabilities of individual data points belonging to
### different clusters.
### one can use the last w and probX in the iterative EM procedure above,
### since the new means and the old means are very close.
### but I would rather calculate w and probX again using new means and covariance matrices
for i in range(n):
    for j in range(k):
        ### My multivariate normal pdf function
        uSu = np.dot( np.dot(iris[i,range(dim)]-mius[:,j], np.linalg.inv(sigmas[j,:])), \
                             iris[i,range(dim)]-mius[:,j])/2.0
        mvnPDF = np.exp(-uSu)/np.sqrt(np.power(2*np.pi, dim)*np.linalg.det(sigmas[j,:]))
        probX[i,0] = probX[i,0] + mvnPDF*probs[j,0] ### probs are the new probs
                                                    ### since I always update probs in the iterative procedure above

for i in range(n):
    for j in range(k):
        ### My multivariate normal pdf function
        uSu = np.dot( np.dot(iris[i,range(dim)]-mius[:,j], np.linalg.inv(sigmas[j,:])), \
                             iris[i,range(dim)]-mius[:,j])/2.0
        mvnPDF = np.exp(-uSu)/np.sqrt(np.power(2*np.pi, dim)*np.linalg.det(sigmas[j,:]))
        w[j,i] = mvnPDF*probs[j,0]/probX[i,0] ### calculate the weight (posterior) w
                                              ### use largest in w[:,i] for cluster label for data point i

### find membership of each data point, using probabilities P(C_j|x_i), i.e. w[j,i]
membership = [1]*n
for i in range(n):
    membership[i] = np.argsort(w[:,i])[k-1]  ### find the max probability to decide membership
    
print 'Cluster Membership:'
sizes = [0]*k  ### sizes of clusters
for i in range(k):
    ### calculate sizes of cluster
    for j in range(n):
        if membership[j]==inds[i]: ### if membership of current data point is the current cluster
            sizes[i] = sizes[i]+1  ### the size of current cluster +1, cluster is in order of ascending norms
    
    ### find members of each cluster
    clusInd = [0]*sizes[i] ### save indices of data points for each cluster
    ind = 0
    for j in range(n):
        if membership[j]==inds[i]: ### if membership of current data point is the current cluster
            clusInd[ind] = j       ### put the current data point in the cluster,
            ind = ind+1            ### move the position in current cluster to next position

    outstr = ''  ### for output format, using a string to save output
    for j in range(sizes[i]):
        outstr = outstr+str(clusInd[j])+',' ### format output
    print outstr[0:(len(outstr)-1)] ### print membership, no last ','

### print sizes of clusters
outstr = ''  ### for output format, using a string to save output
for j in range(k):
    outstr = outstr+str(sizes[j])+' ' ### format output
print 'Size: %s' % outstr[0:(len(outstr)-1)] ### print output, no last ','

if (sys.argv[3] == 'Y') | (sys.argv[3] == 'y'):
    ### read in true labels
    f = open(sys.argv[1], 'r')
    labels = [' ']*n
    ind = 0
    for line in f:
        if line != '\n':
            labels[ind] = line.split(',')[dim] ### get labels, last column in iris data
            ind = ind+1
    f.close()

    ### find unique labels, so we can calculate purity
    ### you can use np.unique
    uniqLabels = [labels[0]]
    for i in range(1,n):
        new_ = True
        for j in range(len(uniqLabels)):
            if (uniqLabels[j] == labels[i]): ### if the current data lable is in the uniqLabels,
                new_ = False  ### its label is not new, set new_ to false, so not add to uniqLabels
                break
        if (new_): ### if the lable of current data point is new, add its lable in uniqLabels
            uniqLabels.append(labels[i])

    ns = np.zeros([k, len(uniqLabels)]) ### save all n_ij s', row is EM prediction, column is true lable

    for i in range(n):
        ind1 = membership[i] ### predicted lable using EM
        tLab = labels[i] ### true lable from data
        ind2 = 0
        for j in range(len(uniqLabels)): ### find which uniqLabel is the current lable, index in uniqLabels
            if tLab == uniqLabels[j]:
                ind2 = j
                break
        ns[ind1, ind2] = ns[ind1, ind2]+1 ### add one to the entry in ns[ind1, ind2]

    purity = 0 ### calculate purity
    for i in range(k):
        purity = purity+max(ns[i,]) ### calculate purity of each cluster
    print 'Purity: %.3f' % (float(purity)/150) ### print purity

else:
    print 'No label is provided, program finishes.'
