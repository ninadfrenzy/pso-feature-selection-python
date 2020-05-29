import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
alpha_const = 0.9
data = pd.read_csv('train.csv')
data = data.sample(n=100)
X = data.iloc[:,:-1].to_numpy()
Y = data.iloc[:,-1].to_numpy()
classifier = LogisticRegression(solver='liblinear',multi_class='ovr')
def load_dataset_pandas():
    dataset = pd.read_csv('train.csv')
    return dataset
def f_per_particle(m, alpha):
    total_features = 20
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j
def f(x, alpha=alpha_const):
    
    n_particles = x.shape[0] #total rows
    
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)
def sns_plot_fig():
    # data = pd.read_csv('train.csv')
    # data = data.sample(n=100)
    # plotfig = sns.pairplot(data, hue='price_range')
    # plotfig = plotfig.fig
    # return plotfig
    data = pd.read_csv('train.csv')
    data = data.sample(n=100)
    sns.pairplot(data, hue='price_range')
def PSO(num_particles=100, alpha=0.9, iterations=100):
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
    global alpha_const
    alpha_const = alpha
    # Call instance of PSO
    dimensions = 20 # dimensions should be the number of features
    # optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
    # optimizer.reset()
    optimizer = ps.discrete.BinaryPSO(n_particles=num_particles, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=iterations)
    print('final cost is ', cost)
    print('final pos is ', pos)
    xcols = data.columns.tolist()
    xcols = xcols[:-1]
    sub_cols = list()
    for i in range(len(pos)):
        if(pos[i] == 0):
            sub_cols.append(xcols[i])
    X_selected_features = X[:,pos==1]
    classifier.fit(X_selected_features, Y)
    predY = classifier.predict(X_selected_features)
    sub_acc = accuracy_score(Y, predY)
    return sub_cols, sub_acc
def classification():
    data = pd.read_csv('train.csv')
    # data = data.sample(n=100)
    
    X = data.iloc[:,:-1].to_numpy()
    Y = data.iloc[:,-1].to_numpy()
    classifier = LogisticRegression(solver='liblinear',multi_class='ovr')
    classifier.fit(X, Y)
    predY = classifier.predict(X)
    main_acc = accuracy_score(Y, predY)
    return main_acc
