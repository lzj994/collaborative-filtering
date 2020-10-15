import numpy as np
import scipy.sparse as sp
import scipy.stats as spstats
import random
import fctns
from sklearn.metrics import average_precision_score
import time
import LowRankALS as LRA
num_users = 301
num_items = 4001
############################
	##Tuning Parameters##
############################
Lambda = 250
sigma = 10**(-8)
alpha = 40
num_iter = 20 #sweeps
f = 30#lower rank that will be used to predict matrix


#Read train and test data, and create R_train and R_test in CSR sparse format
def read_file(filename):
	raw_data = np.genfromtxt(filename, dtype=np.int32)
	users = raw_data[:, 0] #row
	items = raw_data[:, 1] #column
	ratings = raw_data[:, 2].astype(np.float64)#data
	R = sp.coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
	return R.tocsc()#compressed sparse column format

R_train = read_file('train4.txt')
R_test = read_file('test4.txt') 
'''Rorg=R_train+R_test
R_Test=sp.csr_matrix(np.zeros((81,2217)))
R_Train=sp.csr_matrix(np.zeros((81,2217)))
R_Test[60:,1700:]=Rorg[60:,1700:]
R_Train[:60,:1700]=Rorg[:60,:1700]
R_test=R_Test
R_train=R_Train'''
#####create preference-matrix#####
P = np.zeros((num_users,num_items),int,'F')#store elements in column memory,num_users-by-num_items
P[R_train.nonzero()] = 1

#####create confidence-matrix#####
#C = 1+alpha*R_train.toarray('F')#need to add one to all entries
C = 1+alpha*np.log(1+(R_train.toarray('F')/sigma))
#create (Cui) * Pui,
A = C*P

##################################
		#Randomized SVD#
##################################
#want to approximate the matrix R using Randomized SVD to minimize computations

#############STAGE A##############
#low_rank = 10
##create random matrix omega & compute Y = A * omega
#Y = A.dot(np.random.rand(num_items, low_rank))
##perform QR factorization on Y
#Q , R = np.linalg.qr(Y,'reduced')

#############STAGE B##############
#compute B = Q.T * A
#B = Q.T.dot(A)
##perform SVD on matrix B
#U_tilda, S , V_T = np.linalg.svd(B)
##Compute U to get A = Q * B = Q * U_tilda * S * (V_T) = U*S*(V_T)
#U = Q.dot(U_tilda) 

##################################
		#Initialize X,Y#
################################## 

# Initialize X (f-by-m) and Y (f-by-n)
X = np.zeros((f, num_users), order='F')
Y = np.zeros((f, num_items), order='F')
X[:] = np.random.randn(f, num_users)
Y[:] = np.random.randn(f, num_items)

##################################
		#Train X,Y#
##################################
#want to train X and Y against three different matrices 
#the three matrices are the A-confidence*preference matrix,
#the C-confidence Matrix, and the P-preference matrix

t0 = time.time()
#fctns.ALS_train(A, X, Y, C, Lambda, num_iter)
LRA.ALS_train(A,X,Y,C,Lambda,num_iter)
t1=time.time()
print "training time: ",t1 - t0


##################################
	#Evaluate Predictions#
##################################

#the predicted preference will be P_hat = (X.T)Y
P_hat = (X.T).dot(Y)

#rank_tilda[u,i] = R_test[u,i]*rank[u,i]  
indicies=R_test.nonzero() 

rank_tilda = 0
for u,i in zip(indicies[0],indicies[1]):
	rank_tilda += R_test[u,i]*(1-(spstats.percentileofscore(P_hat[u,:],P_hat[u,i],'weak'))/100)

R_test_sum = np.sum(R_test)

print rank_tilda/R_test_sum
'''
#popularity model
rank_tilda = 0
pop=np.zeros(num_items)
for i in range(num_items):
    pop[i]=np.sum(R_train[:,i])
for u,i in zip(indicies[0],indicies[1]):
    rank_tilda += R_test[u,i]*(1-(spstats.percentileofscore(pop,pop[i],'weak'))/100)
print rank_tilda/R_test_sum
'''
