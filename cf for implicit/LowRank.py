import random
import math
import numpy as np

def LowRank_stageA(A, r, p, delta):
	# A is a m*n matrix, for which we want the lower rank approximation
	# r: our confidence level for the error estimte is 1-10^(-r)
	# p: when singular values decay very slowly, we pre-multiply A by (A*A.T)^p
	# Q: the output, a ON matrix whose range approximates the range of A
	m, n = A.shape
	# draw r columns of standard Gaussian vectors
	w = np.random.normal(0, 1, (n, r))
	Y = np.linalg.matrix_power(A.dot(A.T), p).dot(A).dot(w)
	j = 0
	Q = np.zeros((m, 0))
	Y_norm = np.linalg.norm(Y, axis = 0)
	Ymax = delta
	# stop when the error is small enough
	while (Ymax > delta/(10*math.sqrt(2/math.pi))):
		j = j+1
		# projection and normalization
		Y[:, j-1] = (np.identity(m) - Q.dot(Q.T)).dot(Y[:, j-1])
		q = Y[:, j-1] / np.linalg.norm(Y[:, j-1])
		Q = np.insert(Q, j-1, q.T, axis =1 )   # insert the q column to the end of Q
		w = np.random.normal(0 , 1, (n, 1))  
		y = (np.identity(m) - Q.dot(Q.T)).dot(np.linalg.matrix_power(A.dot(A.T), p)).dot(A).dot(w)
		Y = np.insert(Y, j+r-1, y.T, axis =1)   #i nsert the new test column to the end of Y matrix
		for i in xrange(j, j+r-1):
			k = np.sum(np.multiply(Q[:,j-1], Y[:,i]))
			Y[:,i] = Y[:,i] - Q[:,j-1]*k
		# find the y column with the largest norm
		Y_norm = np.linalg.norm(Y, axis = 0)
		Ymax = max(Y_norm[j:j+r])
	return Q

def LowRank_stageB(A, Q):
	# A is a m*n matrix, for which we want the lower rank approximation
	# Q: the matrix we get from stage A
	B = Q.T.dot(A)
	U, s, V = np.linalg.svd(B, full_matrices = False)
	U = Q.dot(U)
	return U, np.diag(s),V





