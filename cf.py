import random
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# in the comments, m refers to number of users, n refers to number of items

def compute_rmse(R, prediction):
	# Compute RMSE based on a matrix filled with predicted ratings
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# prediction: Predicted ratings for all pairs of users and items, a dense matrix of size m-by-n
	# Return value -- RMSE
	# TODO
    #a,b=R.nonzero()
    indicies=R.nonzero()
    omiga=indicies[0].shape[0]
    #print omiga
    m,n=R.shape[0],R.shape[1]
    C=np.zeros((m,n))
    C[indicies]=prediction[indicies]
    RMSE=(np.linalg.norm(C-R))/np.sqrt(omiga)
    return RMSE

def compute_rmse_UV(R, U, V):
	# Compute RMSE based on U and V
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Return value -- RMSE
	# TODO
    #a,b=R.nonzero()
    indicies=R.nonzero()
    omiga=indicies[0].shape[0]
    UtV=U.T.dot(V)
    m,n=R.shape[0],R.shape[1]
    C=np.zeros((m,n))
    C[indicies]=UtV[indicies]    
    #for (i,j) in zip(a,b):
        #C[i,j]=np.dot(U[:,i],V[:,j])
    RMSE=splinalg.norm(sp.csr_matrix(C)-R)/np.sqrt(omiga)
    return RMSE

def compute_cost_func(R, U, V, Lambda):
	# Compute the cost function in ALS (see details in homework instructions)
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Return value -- Value of the cost function
	# TODO
     m,n=R.shape[0],R.shape[1]
     indicies=R.nonzero()
     C=np.zeros((m,n))
     UtV=U.T.dot(V)
     C[indicies]=R[indicies]-UtV[indicies]
     cost=(np.linalg.norm(C))**2+Lambda*((np.linalg.norm(U))**2+(np.linalg.norm(V))**2)
     return cost

def ALS_train(R, U, V, Lambda, num_iter):
	# Optimize the cost function and rewrite U and V
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# num_iter: Number of outer iterations to run
	# (No return value)
	# TODO
     m,n=R.shape[0],R.shape[1]
     for k in range(num_iter):
         for i in range(n):
             a=R[:,i].nonzero()[0]
             #a=R[:,i].nonzero()
             #R_bar=R[a,i]
             R_bar=R[a,i].toarray()
             U_bar=U[:,a]
             A=U_bar.dot(U_bar.T)+Lambda*np.eye(V[:,i].shape[0])
             #y=U_bar.dot(np.asarray(R_bar.todense())[:,0])
             y=U_bar.dot(R_bar)
             #V[:,i] =np.linalg.solve(A,y)
             V[:,i]=(np.linalg.inv(A).dot(y))[:,0]
             #print np.allclose(A.dot(V[:,i]),y)
         for j in range(m):
             #a=R[j,:].nonzero()[0]
             b=R[j,:].nonzero()[1]
             #R_bar=R[j,b]
             R_bar=R[j,b].toarray()
             V_bar=V[:,b]
             B=V_bar.dot(V_bar.T)+Lambda*np.eye(U[:,j].shape[0])
             z=(V_bar).dot(R_bar.T)
             #z=(V_bar).dot(np.asarray(R_bar.T.todense())[:,0])
             #print np.shape(z)
             #U[:,j]=np.linalg.solve(B,z)
             U[:,j]=np.linalg.inv(B).dot(z)[:,0]
             #np.linalg.norm(B.dot(U[:,j])-z)
         print "iteration: ",k+1," ", compute_cost_func(R,U,V,Lambda)
     return

def ALS_predict(R, U, V, Lambda, personal_items, personal_ratings):
	# Based on the model you've trained,
	# predict the ratings for all the items given your own ratings
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# personal_items: Indices of the items in your own ratings
	# personal_ratings: Your own ratings, corresponding to the indices in 'personal_items'
	# Return value: A n-vector containing the predicted ratings
	# TOD
 #I create a new sparse matrix with the first row my own rating,
# and rest of rows are rows of R_train
     k, num_items = V.shape
     RR=np.zeros((R.shape[0]+1,R.shape[1]))
     RR=sp.csr_matrix(RR)
     RR[1:,:]=R
     u=np.zeros(R.shape[1])
     u[personal_items]=personal_ratings
     u=sp.csr_matrix(u)
     RR[0,:]=u
     U = np.zeros((k, R.shape[0]+1), order='F')
     V = np.zeros((k, R.shape[1]), order='F')
     U[:] = np.random.randn(k, R.shape[0]+1)
     V[:] = np.random.randn(k, R.shape[1])
     ALS_train(RR,U,V,Lambda,20)
     v=U[:,0]
     v=v.T.dot(V)
     return v
