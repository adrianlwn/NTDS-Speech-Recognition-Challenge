import numpy as np
import pandas as pd
from scipy import sparse, spatial
from tqdm import tqdm

# pipeline function for semisupervised learning using graphs
def semisup_test_all_dataset(features_og, label_vec, train_x, train_y, test_x, batch_size, NEIGHBORS, alpha, beta):
    """Test semisupervised graph learning algorithm for entire dataset.
			- features_og : original copy of all MFCCs
			- train_x : indices of training samples in dataset
			- train_y : labels of training samples
			- test_x : indices of testing samples in dataset
			- batch_size : number of samples to be predict per iteration in main loop
			- NEIGHBORS : number of neirest neighbors in k-NN sparsification
			- alpha : hyper-parameter which controls the trade-off between the data fidelity term and the smoothness prio
			- beta : 
				"""

		# number of batches to loop through
		n_batch = int(len(test_x) / batch_size)

		# encode training samples classes into 1-hot array
		n_class = np.max(train_y)
		Y = np.eye(len(class_names))[train_y - 1].T

		# accumulate accuracy values
		accuracy_tot = []
		accuracy = []
		remaining_test = np.array(test_x)

		for batch in tqdm(range(n_batch)):
		
			# get batch indices
			potential_elements  = np.array(list(enumerate(remaining_test)))
			indices = np.random.choice(potential_elements[:,0].reshape(-1,), batch_size, replace=False)
			batch_index = potential_elements[:,0].reshape(-1,)[indices]
			remaining_test = np.delete(remaining_test, indices)
				
			# build graph
			features = pd.DataFrame(features_og['mfcc'], np.append(train_x, batch_index))
			features -= features.mean(axis=0)
			features /= features.std(axis=0)

			distances = spatial.distance.squareform(spatial.distance.pdist(features,'cosine'))
			
			n=distances.shape[0]
			kernel_width = distances.mean()
			weights = np.exp(np.divide(-np.square(distances),kernel_width**2))
			np.fill_diagonal(weights,0)

			# k-NN sparsification
			for i in range(weights.shape[0]):
				idx = weights[i,:].argsort()[:-NEIGHBORS]
				weights[i,idx] = 0
				weights[idx,i] = 0
			
			# compute laplacian
			degrees = np.sum(weights,axis=0)
			laplacian = np.diag(degrees**-0.5) @ (np.diag(degrees) - weights) @ np.diag(degrees**-0.5)
			laplacian = sparse.csr_matrix(laplacian)

			# add test samples to 1-hot array
			M = np.zeros((len(class_names), len(train_y) + batch_size)) # mask matrix
			M[:len(train_y),:len(train_y)] = 1
			Y_compr = np.concatenate((Y, np.zeros((len(class_names), batch_size))), axis=1)
			y = np.concatenate((train_y,np.zeros((batch_size,))))

			# Solve
			X = solve(Y_compr, M, laplacian, alpha = alpha, beta = beta)

			# Make label vector
			x_hat = np.argmax(X,axis = 0) + np.ones(X[0,:].shape)

			# Unify labels 13-30
			x_hat_adapted = adapt_labels(x_hat)
			true_y = np.concatenate((train_y,label_vec[batch_index]))
			y_adapted = adapt_labels(true_y)

			# Only consider unknowns
			accuracy_tot.append(np.sum(x_hat[(len(x_hat)-batch_size):]==true_y[(len(x_hat)-batch_size):])/batch_size) # for all 30 words
			accuracy.append(np.sum(x_hat_adapted[(len(x_hat)-batch_size):]==y_adapted[(len(x_hat)-batch_size):])/batch_size) # only core words
			
		return accuracy, accuracy_tot
