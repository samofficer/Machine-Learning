import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots, show
from statsmodels.api import OLS
from statsmodels.api import add_constant
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial
from sklearn.pipeline import Pipeline # step-by-step recipe
from sklearn.decomposition import PCA # It reduces the number of features by combining them into new ones (called principal components) that keep the most important information.
from sklearn.cross_decomposition import PLSRegression #Partial Least Squares Regression, selects variables explain variance
from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)
from l0bnb import fit_path # does subset selection (finds the path)


df = pd.read_stata("Causal ML/code files/dataverse_files/AER merged.dta") #specify path
# Dependent Variable
Y_cols = "amount"  
# Independent Variables
X_cols = [
    "hpa","freq","mrm2","years","dormant","female","couple",
    "median_hhincome","psch_atlstba","powner","pop_propurban",
    "ave_hh_sz","page18_39"
]
# turn Y string into list so we can add
regression_cols = [Y_cols, "treatment"] + X_cols
data = df[regression_cols]
# get rid of na's
data = data.dropna()
# separate outcome and predictors
Y=data[Y_cols]
# .copy() make separate copy of selected columns, leave data unchanged
X=data[["treatment"] + X_cols].copy()
# add interaction terms with treatment, make new column for each interaction
# inside X. LHS makes new column in X with name, RHS is interaction
for col in X_cols:
    X["treatment_x_" + col] = data["treatment"] * data[col]

# add intercept
X = add_constant(X)

# run OLS controlling for possible heteroskedasticity in HPA
Model1 = OLS(Y,X).fit(cov_type="HC1")
# export as latex
# latex_output = Model1.summary().as_latex()
# with open("OLS_results.tex", "w") as f:
    # f.write(latex_output)

# ridge makes its own intercept, axis=1 tells pandas to remove a column not row
# removing const column we made before from our X
D = X.drop('const', axis=1)

# convert the pandas table into a NumPy array, a grid of numbers without column labels
X_ridge = np.asarray(D)

# standardize predictors, mean(0) calculates mean down the rows, giving one average per column
# [None,:] puts the averages into a single row array. None adds a dimension and : selects all the avgs
# Subtraction applies each column's average to every value in the column
Xs = X_ridge - X_ridge.mean(0)[None, :]
# calc each std per column
X_scale = X_ridge.std(0)
# divide each centered column by its own std
Xs = Xs / X_scale[None, :]

# linspace creates 100 evenly spaced numbers starting at 8 ending at -2
# 8 and 2 exps create a massive range of penalties
# divide by y std to adjust grids overall scale
# stores the 100 penalties under lambdas
lambdas = 10**np.linspace(8, -2, 100) / Y.std()

soln_array = skl.ElasticNet.path(Xs,# standardized predictors
                                 Y, # donation amounts
                                 l1_ratio=0, # ridge penalty, lasso is 1
                                 alphas=lambdas, # use the penalties we made
                                 max_iter=100000, # more iterations to find coefs, 100k passes through the coefs
                                 tol=1e-8 # more accurate solution before stopping, check whether future updates are sufficiently small
                                 # and its solution is sufficiently accurate
                                 )[1] # selects second result

soln_path = pd.DataFrame(soln_array.T, # transpose array, pd.dataframe turns that array into a pandas
                         # table with labels
                         columns=D.columns, # D.columns supplies predictors names
                         index=-np.log(lambdas)) # sets the row labels to the negative nat log of each penalty
soln_path.index.name = 'negative log(lambda)'
# print(soln_path)

# create coef plot, subplots creates figure with a plotting area
# figsize 8 inches wide and tall
# path fig refers to whole figure, ax is plotting area where we draw lines and set labels
path_fig, ax = subplots(figsize=(8,8))
# .plot draws one line for each column, one per coef
# ax=ax tells pandas to draw on plotting area we just created
soln_path.plot(ax=ax, legend=False)

# set the axis labels and text size, $$ notation formats as math
# r keeps python from treating backslashes as escape chars
ax.set_xlabel(r'$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)

# labels lines with predictor names and puts legend upper left
ax.legend(loc='upper left')

# save figure to same file to add to latex doc
path_fig.savefig(
    r"C:\Users\Admin\OneDrive - University of Toronto\School\Causal ML\code files\ridge_coefficients.pdf",
    bbox_inches="tight"
)

# ridge cross validation
K = 5

kfold = skm.KFold(n_splits=K, # request 5 folds
                  random_state=0, # make the split reproducible so rerunning gives same folds
                  shuffle=True # mixes observations before dividing so folds dont follow row order
                  )
ridge = skl.ElasticNet(l1_ratio=0, # ridge
                       max_iter=100000,
                       tol=1e-8)

# define standardization step
# subtract each columns mean, divide by its std, this is just the same as the manual one above
scaler = StandardScaler(with_mean=True, with_std=True)

pipe = Pipeline(steps=[  # steps of operation, scale/standardize and give name scalar
    ('scaler', scaler),
    ('ridge', ridge) # assign ridge model to 'ridge' after
])

# dictionary pairing param name with possible values, ridge is name of ridge step in pipeline
#__ connects step to one of its settings, alpha is penalty setting
# lambdas is 100 penalty values created
param_grid = {'ridge__alpha': lambdas} 
# gridsearch chooses the lambda param
grid = skm.GridSearchCV(pipe,  # standardize predictors then fit ridge
                        param_grid,  # repeat process for every penalty
                        cv=kfold, # use k fold
                        scoring='neg_mean_squared_error') # assess predictions using neg MSE

# performs the search: 100 penalties * 5 folds = 500 fits, auto refits selected model on supplied data
print("Starting cross-validation...", flush=True)
grid.fit(X_ridge,Y)
print("cross-validation done", flush=True)
best_lambda = grid.best_params_['ridge__alpha'] # retrieve penalty with lowest average CV MSE

# set up figure
ridge_fig, ax = subplots(figsize=(8,8))
ax.errorbar(
    -np.log(lambdas), # neg natural log of each penalty
    -grid.cv_results_['mean_test_score'], # average validation MSE at each penalty
    yerr=grid.cv_results_['std_test_score']/np.sqrt(K) # how much validation varies over 5 folds, divide to get
    # approx SE. yett draws bars extending that amount above and below the avg
)

ax.set_xlabel(r'$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)
print("Selected lambda:", best_lambda)
print("lowest average CV MSE:", -grid.best_score_)

ridge_fig.savefig(
    r"C:\Users\Admin\OneDrive - University of Toronto\School\Causal ML\code files\ridge_MSECV.pdf",
    bbox_inches="tight"
)

lassoCV = skl.ElasticNetCV(alphas=100,
                           l1_ratio=1,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scalar', scaler),
                         ('lasso', lassoCV)])
pipeCV.fit(X_ridge, Y)
tuned_lasso = pipeCV.named_steps['lasso'] # named steps lets you access pipeline step by its name

lambdas, soln_array = skl.Lasso.path(Xs, # standardized predictors
                                     Y, # donation amounts
                                     l1_ratio=1, # lasso model
                                     alphas=100)[:2] # 100 penalty values, [:2] keeps first 2 values

soln_path = pd.DataFrame(soln_array.T,
                         columns=D.columns,
                         index=-np.log(lambdas))

# plot coefs like ridge
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel(r'$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20);

# plot CV error plot like ridge
lassoCV_fig, ax = subplots(figsize=(8, 8))

ax.errorbar(
    -np.log(tuned_lasso.alphas_),
    tuned_lasso.mse_path_.mean(1),
    yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K)
)

# alpha_ singular the one selected penalty, alphas_ would be plural
# c = 'k' means black, ls= '--' means dashed
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')

ax.set_xlabel(r'$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)

path_fig.savefig(
    r"C:\Users\Admin\OneDrive - University of Toronto\School\Causal ML\code files\lasso_coefficients.pdf",
    bbox_inches="tight"
)

lassoCV_fig.savefig(
    r"C:\Users\Admin\OneDrive - University of Toronto\School\Causal ML\code files\lasso_MSECV.pdf",
    bbox_inches="tight"
)

show()