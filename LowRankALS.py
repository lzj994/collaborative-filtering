import random
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import LowRank

def ALS_train(R, X, Y, C, Lambda, num_iter):
	# Optimize the cost function and rewrite X and Y
	# R: Rating matrix (m-by-n), a dense matrix
	# X: f-by-m dense matrix
	# Y: f-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# num_iter: Number of outer iterations to run
	# (accessing matrices column wise)
	f,n = Y.shape 
	m,n = R.shape
	Lambda_matrix = Lambda*np.identity(f)
	C = C-1
	Q = LowRank.LowRank_stageA(R, 8, 0, 8000)
	U,S,V = LowRank.LowRank_stageB(R,Q) 
	print "Low Rank is", Q.shape[1]
	for k in range(num_iter):
		B = X.dot(U).dot(S).dot(V)
		X_tilda = X.dot(X.T) + Lambda_matrix#f-by-f matrix
		for i in range(n):
			#nonzero1 = R[:,i].nonzero()[0]#nonzero entries along the ith column of R
			X_hat = X.dot(C[:,i].reshape(m,1)*X.T)#f-by-f matrix
			A = X_tilda + X_hat #f-by-f matrix
			#x_bar = X[:,nonzero1]#f-by-omega matrix 
			#r_hat = R[nonzero1,i]#omega-by-1 matrix, only nonzero entries
			#b = x_bar.dot(r_hat).reshape(f,1)#f-by-1 matrix
			b = B[:,i]
			Y[:,i]=splinalg.cgs(A,b)[0]
			
		B = Y.dot(V.T).dot(S).dot(U.T)
		Y_tilda = Y.dot(Y.T) + Lambda_matrix#f-by-f matrix
		for j in range(m):	
			nonzero1 = R.T[:,j].nonzero()[0]#nonzero entries along the ith row of R
			Y_hat = Y.dot((C.T[:,j].reshape(n,1))*Y.T) #f-by-f matrix
			A = Y_tilda + Y_hat #f-by-f matrix
			#y_tilda = Y[:,nonzero1]#f-by-omega matrix
			#r_tilda = R.T[nonzero1, j]#omega-by-1 matrix, observed ratings for each user
			#b = y_tilda.dot(r_tilda).reshape(f,1)#f-by-1 matrix
			b = B[:,j]
			X[:,j] = splinalg.cgs(A,b)[0]
		
	return
	
def ALS_train2(R, X, Y, Lambda, num_iter):

	# Optimize the cost function and rewrite X and Y
	# R: Rating matrix (m-by-n), a dense matrix 
	# X: f-by-m dense matrix
	# Y: f-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# num_iter: Number of outer iterations to run
	# (accessing matrices column wise)
	f,n = Y.shape 

	Lambda_matrix = Lambda*np.identity(f)
	for k in range(num_iter):
	
		Y = np.linalg.solve(X.dot(X.T) + Lambda_matrix, X.dot(R))
		
		X = np.linalg.solve(Y.dot(Y.T) + Lambda_matrix, Y.dot(R.T))
		
	return