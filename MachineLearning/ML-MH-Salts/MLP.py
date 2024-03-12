import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastdataing as fd
from tqdm import tqdm
from data import read_data, removing_duplicates, feature_selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Integer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from joblib import dump, load
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore")

def evaluate_model(params):
    hidden_layers, neurons_per_layer = params
    hidden_layer_sizes = tuple([neurons_per_layer] * hidden_layers)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
    model.fit(X_train, y_train)
    return -model.score(X_train, y_train)

def nfoldCV(X_train,y_train,n=5,search_type ="random"):

	mlp = MLPRegressor()
	max_neurons = 50
	param_grid = {
		'hidden_layer_sizes': [(n,) for n in range(1, max_neurons+1)]+ \
		[(n, m) for n in range(1, max_neurons+1) for m in range(1, max_neurons+1)],#+ \
		# [(n, m, q) for n in range(1, max_neurons+1) for m in range(1, max_neurons+1) for q in range(1, max_neurons+1)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'solver': ['lbfgs', 'sgd', 'adam'],
		'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': [0.001, 0.01, 0.1],
		'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
	}
	scoring = 'r2' #'neg_root_mean_squared_error'
	if search_type == "grid":
		# 创建GridSearchCV对象
		opt = GridSearchCV(mlp, param_grid, cv=n,verbose=3,scoring=scoring)
		opt.fit(X_train, y_train)
		print(f"Best parameters ({search_type}):",opt.best_params_)
		print("Best parameters score:",opt.best_score_)
	elif search_type == "random":
		# 创建RandomizedSearchCV对象
		opt = RandomizedSearchCV(mlp, param_grid, n_iter=100,cv=n,verbose=3,scoring=scoring)
		opt.fit(X_train, y_train)
		print(f"Best parameters ({search_type}):",opt.best_params_)
		print("Best parameters score:",opt.best_score_)
	elif search_type == "bayes":
		# 创建BayesSearchCV对象
		# param_grid.pop("hidden_layer_sizes")
		param_grid['hidden_layer_sizes'] = [5,10,15,20,25]
		opt = BayesSearchCV(mlp, param_grid, n_iter=50,cv=n,verbose=3,scoring=scoring)
		print(opt)
		opt.fit(X_train, y_train)
		print(f"Best parameters ({search_type}):",opt.best_params_)
		print("Best parameters score:",opt.best_score_)

	elif search_type == "bayes_gpm":
		search_space = [Integer(1,3, name='hidden_layers'), Integer(1,11, name='neurons_per_layer')]
		opt = gp_minimize(evaluate_model,search_space,verbose=True)
		print(f"Best parameters ({search_type}):",opt.x[0],opt.x[1])
		print("Best parameters score:",-opt.fun)

	return opt


def train_model(solver='adam', learning_rate='constant',learning_rate_init=0.01, 
	hidden_layer_sizes=(10, 10), activation="relu",alpha=0.00001,max_iter=1000,
	random_state=2024):

	mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
					   random_state=random_state, 
					   max_iter=max_iter,
					   solver = solver,
					   activation = activation,
					   learning_rate = learning_rate,
					   learning_rate_init = learning_rate_init,
					   alpha = alpha,
					   batch_size=20,early_stopping=True,
					  )
	std_clf = make_pipeline(StandardScaler(), PCA(n_components=None), mlp)
	std_clf.fit(X_train, y_train)
	y_predict = std_clf.predict(X_train)
	r2 = std_clf.score(X_train,y_train)
	print(f"R^2 (train) = {r2}")
	y_predict_val = std_clf.predict(X_val)
	r2_val = std_clf.score(X_val,y_val)
	print(f"R^2 (val) = {r2_val}")
	
	dump(std_clf, "./data/MLP_model.pkl")

	return y_train, y_predict, y_val, y_predict_val

def load_model():

	std_clf = load("./data/MLP_model.pkl")

	y_predict = std_clf.predict(X_train)
	r2 = std_clf.score(X_train,y_train)
	print(f"R^2 (train) = {r2}")
	y_predict_val = std_clf.predict(X_val)
	r2_val = std_clf.score(X_val,y_val)
	print(f"R^2 (val) = {r2_val}")

	return y_train, y_predict, y_val, y_predict_val

def predict_model(X_test,y_test):

	std_clf = load("./data/MLP_model.pkl")

	y_predict_test = std_clf.predict(X_test)
	r2 = std_clf.score(X_test,y_test)
	print(f"R^2 (test) = {r2}")
	return y_test, y_predict_test

def plot_figs(y_train, y_predict, y_val, y_predict_val):
	fig = fd.add_fig(figsize=(10,8),size=16)
	plt.subplots_adjust(wspace=0.3)

	ax = fd.add_ax(fig,subplot=(2,2,1))
	ax.scatter(y_train,y_predict,edgecolors=(0, 0, 0))
	ax.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()],'r--',lw=2)
	ax.set_xlim(y_train.min(),y_train.max())
	ax.set_ylim(y_train.min(),y_train.max())
	ax.set_xlabel("Experimental T (K)")
	ax.set_ylabel("MLP Predicted (train) T (K)")

	ay = fd.add_ax(fig,subplot=(2,2,2))
	ay.scatter(y_train,y_predict-y_train,edgecolors=(0, 0, 0))
	ay.plot([y_train.min(),y_train.max()],[0,0],'r--',lw=2)
	ay.set_xlim(y_train.min(),y_train.max())
	ay.set_ylim(-15,15)
	ay.set_xlabel("Experimental T (K)")
	ay.set_ylabel(r"$\regular \it Res_i$ (train) (K)")
	# ----------- test ------------
	az = fd.add_ax(fig,subplot=(2,2,3))
	az.scatter(y_val,y_predict_val,edgecolors=(0, 0, 0))
	az.plot([y_val.min(),y_val.max()],[y_val.min(),y_val.max()],'r--',lw=2)
	az.set_xlim(y_val.min(),y_val.max())
	az.set_ylim(y_val.min(),y_val.max())
	az.set_xlabel("Experimental T (K)")
	az.set_ylabel("MLP Predicted (test) T (K)")

	aa = fd.add_ax(fig,subplot=(2,2,4))
	aa.scatter(y_val,y_predict_val-y_val,edgecolors=(0, 0, 0))
	aa.plot([y_val.min(),y_val.max()],[0,0],'r--',lw=2)
	aa.set_xlim(y_val.min(),y_val.max())
	aa.set_ylim(-15,15)
	aa.set_xlabel("Experimental T (K)")
	aa.set_ylabel(r"$\regular \it Res_i$ (test) (K)")
	plt.savefig("./imgs/MLP.png",dpi=300)
	plt.show()

	return

def plot_fig(y_test, y_predict_test):
	fig = fd.add_fig(figsize=(10,4),size=16)
	plt.subplots_adjust(wspace=0.3,bottom=0.15)

	# ----------- test ------------
	az = fd.add_ax(fig,subplot=(1,2,1))
	az.scatter(y_test,y_predict_test,edgecolors=(0, 0, 0))
	az.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',lw=2)
	az.set_xlim(y_test.min(),y_test.max())
	az.set_ylim(y_test.min(),y_test.max())
	az.set_xlabel("Experimental T (K)")
	az.set_ylabel("MLP Predicted (test) T (K)")

	aa = fd.add_ax(fig,subplot=(1,2,2))
	aa.scatter(y_test,y_predict_test-y_test,edgecolors=(0, 0, 0))
	aa.plot([y_test.min(),y_test.max()],[0,0],'r--',lw=2)
	aa.set_xlim(y_test.min(),y_test.max())
	aa.set_ylim(-15,15)
	aa.set_xlabel("Experimental T (K)")
	aa.set_ylabel(r"$\regular \it Res_i$ (test) (K)")
	plt.savefig("./imgs/MLP_test.png",dpi=300)
	plt.show()

	return


if __name__ == '__main__':
	# ---------- data prepare ----------
	f = "./data/1-s2.0-S1364032122009844-mmc1.xlsx"
	df = read_data(f)
	df = removing_duplicates(df)

	# ---------- split data ----------
	df_new = df[df.iloc[:, 0] != 'MgBr2']
	df_mgbr2 = df[df.iloc[:, 0] == 'MgBr2']
	X, y, X_mgbr2, y_mgbr2 = feature_selection(df_new,df_mgbr2)
	X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=2024)
	# print(X_train.shape,X_val.shape)
	# print(y_train.shape,y_val.shape)
	# ---------- 5-fold Cross-Validation (CV) ----------
	search_type = "random"
	opt = nfoldCV(X_train,y_train,n=5,search_type = search_type)
	if search_type == "bayes_gpm":
		y_train, y_predict, y_val, y_predict_val = train_model(
			hidden_layer_sizes = (opt.x[0],opt.x[1])
			)
	else:
		hidden_layer_sizes=opt.best_params_["hidden_layer_sizes"]
		solver=opt.best_params_["solver"]
		learning_rate = opt.best_params_["learning_rate"]
		learning_rate_init=opt.best_params_["learning_rate_init"]
		activation=opt.best_params_["activation"]
		alpha = opt.best_params_["alpha"]

		y_train, y_predict, y_val, y_predict_val = train_model(
			solver=solver, 
			learning_rate = learning_rate,
			learning_rate_init=learning_rate_init, 
			hidden_layer_sizes=hidden_layer_sizes, 
			activation=activation,
			alpha = alpha,
			)
	# y_train, y_predict, y_val, y_predict_val = load_model()
	y_test, y_predict_test = predict_model(X_mgbr2, y_mgbr2)

	plot_figs(y_train, y_predict, y_val, y_predict_val)
	plot_fig(y_test, y_predict_test)
