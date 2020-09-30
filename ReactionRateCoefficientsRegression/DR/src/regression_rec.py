#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../../../Utilities/')
import plotting
# from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import operator
import itertools
from sklearn import metrics
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load

# To run with all processors
n_jobs = -1

# ... this is useless!
trial = 1

# Import dataset
dataset_T = np.loadtxt("./data/N2-N2/dis/Temperatures.csv")
dataset_k = np.loadtxt("./data/N2-N2/rec/DR_RATES-N2-N2-rec.csv")
dataset = "DR_RATES-N2-N2-rec"

# Input argument: vibrational level
# Lev = '3'
Lev = sys.argv[1]

# Reshape the dataset
x = dataset_T.reshape(-1,1)            # T [K]
y = dataset_k[:,0+int(Lev):1+int(Lev)] # k_DR

# Split dataset between train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

# Scale datasets
# Other scalers are possible, for example:
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

# Save the scaling (this is done for each vibrational level)
dump(sc_x, open('scaler_x_'+dataset+'_'+Lev+'.pkl', 'wb'))
dump(sc_y, open('scaler_y_'+dataset+'_'+Lev+'.pkl', 'wb'))

# Print some info about train and test subsets
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Setup the range of variation for hyper-parameters
# Each regressor has its own parameters, for example:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
#hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',),
                 #'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
                 #'leaf_size': (1, 10, 20, 30, 100, 1000,),
                 #'weights': ('uniform', 'distance',),
                 #'p': (1,2,),}]
hyper_params = [{'max_depth': (1, 2, 3, 4, 5, 6, 7, 8,)}]

# Choose the regressor est (=estimator)
# This should be changed to try different regressors.
#est=neighbors.KNeighborsRegressor()
est = DecisionTreeRegressor()

# Use GridSearch to find the best combination of hyper-parameters
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train the MLA and take the time
t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
# print("kNN complexity and bandwidth selected and model fitted in %.6f s" % runtime)
print(f"dtr complexity and bandwidth selected and model fitted in {runtime} s")

# Get some usefull metrics
train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2  = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

# Retrieve the best parameters
sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

# Write best parameters to file
out_text = '\t'.join(['k-nearest-neighbour',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

# Assign best parameters to new variable to be used later
# best_algorithm = gs.best_params_['algorithm']
# best_n_neighbors = gs.best_params_['n_neighbors']
# best_leaf_size = gs.best_params_['leaf_size']
# best_weights = gs.best_params_['weights']
# best_p = gs.best_params_['p']

best_max_depth = gs.best_params_['max_depth']

outF = open("output.txt", "w")
# print('best_algorithm = ', best_algorithm, file=outF)
# print('best_n_neighbors = ', best_n_neighbors, file=outF)
# print('best_leaf_size = ', best_leaf_size, file=outF)
# print('best_weights = ', best_weights, file=outF)
# print('best_p = ', best_p, file=outF)
print(f'best max_depth = {best_max_depth}', file=outF)
print('R2 score is {}'.format(test_score_r2))

outF.close()

# Construct a regressor with the best parameters
# kn = KNeighborsRegressor(n_neighbors=best_n_neighbors,
                         # algorithm=best_algorithm,
                         # leaf_size=best_leaf_size,
                         # weights=best_weights,
                         # p=best_p)
dtr = DecisionTreeRegressor(max_depth=best_max_depth)

# Re-train with the best parameters
t0 = time.time()
dtr.fit(x_train, y_train.ravel())
dtr_fit = time.time() - t0
# print("kNN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit)
print(f"dtr selected and model fitted in {dtr_fit} s" )

# Fit (predict)
t0 = time.time()
# y_kn = kn.predict(x_test)
y_dtr = dtr.predict(x_test)
# kn_predict = time.time() - t0
dtr_predict = time.time() - t0
# print("kNN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict))
print(f"dtr prediction for {x_test.shape[0]} inputs in {dtr_predict} s")

# Write prediction metrics to file
outF = open("output.txt", "a")
# print("kNN complexity and bandwidth selected and model fitted in %.6f s" % dtr_fit, file=outF)
# print("kNN prediction for %d inputs in %.6f s" % (x_test.shape[0], dtr_predict),file=outF)
print(f"dtr complexity and bandwidth selected and model fitted in {dtr_fit}", file=outF)
print(f"dtr prediction for {x_test.shape[0]} inputs in {dtr_predict} s", file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_dtr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_dtr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_dtr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_dtr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_dtr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_dtr)))

# Unscale the data, to be back to original dataset values
x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_kn_dim = sc_y.inverse_transform(y_dtr)

# Plot (units may be wrong, depending on the coefficient)
plt.scatter(x_test_dim, y_test_dim, s=2, c='k', marker='o', label='Matlab')
# plt.scatter(x_test_dim, y_kn_dim,   s=2, c='r', marker='+', label='k-Nearest Neighbour')
plt.scatter(x_test_dim, y_kn_dim,   s=2, c='r', marker='+', label='decision tree')
#plt.title(''Relaxation term $R_{ci}$ regression')
plt.ylabel('$k_{rec}$ $[m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
#plt.savefig("regression_kNN.eps", dpi=150, crop='false')
plt.savefig(sys.path[1]+"/model/pdf/regression_dtr_"+dataset+"_"+Lev+".pdf", dpi=150)
# plt.show()
# plt.close('all')

# Save the model to disk
dump(gs, sys.path[1]+"/model/model_dtr_"+dataset+"_"+Lev+".sav")
