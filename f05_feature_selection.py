# %% [markdown]
# ## Cargar Dataset

# %%
import multiprocessing
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Models
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Feature Selection
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score

# %%
datasets = ['0_CIC-IDS-2017', '1_UNSW-NB15', '2_NF-UNSW-NB15-v2', '3_CSE-CIC-IDS2018', '4_NSL-KDD']

# %% [markdown]
# # Funciones Utilitarias

# %% [markdown]
# ## Dataset IO

# %%
def execute_feature_selection(dataset):
    warnings.filterwarnings("ignore")

    def load_dataset(dataset):
        folder = f'./datasets_clean/{dataset}'
        X = pd.read_csv(f'{folder}/X.csv')
        y = pd.read_csv(f'{folder}/Y.csv')
        return X,y

    def print_to_results(msg, type = 'a'):
        print(msg)
        folder = 'results_feature_selection'
        file = open(f'./{folder}/{dataset}.txt', type)
        file.write(f'{msg}\n\n')
        file.close()

    # %% [markdown]
    # ## Seleccion de Modelos

    # %% [markdown]
    # ### Seleccion de features

    # %%
    def select_features(model, params):
        grid = fit_grid(model, params)
        best_selector = grid.best_estimator_['select']
        selected_features = X_train.columns[best_selector.get_support()]
        save_results(grid, selected_features)

    # %% [markdown]
    # ### Grid Search

    # %%
    def fit_grid(model, params):
        grid = GridSearchCV(model, params, error_score=0)
        grid.fit(X_train, y_train)    
        return grid

    # %% [markdown]
    # ### Guardar resultados

    # %%
    def save_results(grid, selected_features = None):
        train_accuracy = round(grid.best_score_, 6)
        fit_time = round(grid.cv_results_['mean_fit_time'].mean(), 3)
        score_time = round(grid.cv_results_['mean_score_time'].mean(), 3)
        test_accuracy = round(grid.best_estimator_.score(X_test, y_test), 6)

        result =   (f"Best model: {grid.best_estimator_}\n"
                    f"Train Accuracy: {train_accuracy}\n"
                    f"Average Time to Fit (s): {fit_time}\n"
                    f"Average Time to Score (s): {score_time}\n"
                    f"Test Accuracy: {test_accuracy}")
                    
        if(selected_features is not None):
            result =   (f"{result}"
                        f"\nNumber of features: {len(selected_features)}\n"
                        f"{selected_features}")
                    
        print_to_results(result)

    # %% [markdown]
    # ## Lectura del Dataset

    # %%
    print_to_results(f'Starting Feature Selection at {datetime.now()}', 'w')
    X, y = load_dataset(dataset)

    header = (f'Dataset: {dataset}\n'
            f'X shape: {X.shape}\n'  
            f'y shape: {y.shape}\n' 
            f'y proportions: \n{y.value_counts(normalize=True)}\n')

    print_to_results(header)



    # %% [markdown]
    # ### Preprocesamiento del dataset

    # %% [markdown]
    # #### Eliminar columnas constantes

    # %%
    variance_filter = VarianceThreshold(threshold=0)
    Xcols = X.columns
    X = variance_filter.fit_transform(X)
    Xcols = Xcols[variance_filter.get_support()]
    X = pd.DataFrame(X, columns=Xcols)

    # %% [markdown]
    # #### Reshape Y

    # %%
    y = y.values.ravel()

    # %% [markdown]
    # #### Train Test Split

    # %%
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # %% [markdown]
    # ## Modelos Base

    # %%
    print_to_results("Base models")

    # %% [markdown]
    # ### Entrenamiento de modelos base

    # %%
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier()

    lr_params = {'classifier__C':[0.1, 1, 10]}
    knn_params = {'classifier__n_neighbors': [1, 3, 5]}
    tree_params = {'classifier__max_depth': [11, 13, 15]}
    forest_params = {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [1, 3]}

    # %%
    pipe_lr = Pipeline([('classifier', lr)])
    grid_lr = fit_grid(pipe_lr, lr_params)
    save_results(grid_lr)

    # %%
    pipe_knn = Pipeline([('classifier', knn)])
    grid_knn = fit_grid(pipe_knn, knn_params)
    save_results(grid_knn)

    # %%
    pipe_tree = Pipeline([('classifier', tree)])
    grid_tree = fit_grid(pipe_tree, tree_params)
    save_results(grid_tree)

    # %%
    pipe_forest = Pipeline([('classifier', forest)])
    grid_forest = fit_grid(pipe_forest, forest_params)
    save_results(grid_forest)

    # %% [markdown]
    # ### Eleccion del mejor modelo base

    # %%
    classifiers = [lr, knn, tree, forest]
    params = [lr_params, knn_params, tree_params, forest_params]
    best_scores = [grid_lr.best_score_, grid_knn.best_score_, grid_tree.best_score_, grid_forest.best_score_]


    best_index = np.argmax(best_scores)
    classifier = classifiers[best_index]
    classifier_params = params[best_index]
    print_to_results(f"Best base model: {classifier}")

    # %% [markdown]
    # # Feature Selection

    # %% [markdown]
    # ## Filters

    # %%
    print_to_results("Filters")

    # %% [markdown]
    # ### Correlacion

    # %%
    def correlation(X,y):
        y = y.reshape((y.size,-1))
        np_data = np.concatenate([X,y], axis=1)
        pd_data = pd.DataFrame(np_data)    
        corr = pd_data.corr().abs().iloc[-1]
        corr = corr[:-1]
        return np.array(corr)

    # %%
    print_to_results("Correlacion")

    corr_pipe = Pipeline([('select', SelectKBest(correlation)), 
                        ('classifier', classifier)])

    corr_pipe_params = deepcopy(classifier_params)

    corr_pipe_params.update({'select__k':[10,15,20]})

    select_features(corr_pipe, corr_pipe_params)

    # %% [markdown]
    # ### P - value

    # %%
    print_to_results("P-value")

    p_value_pipe = Pipeline([('select', SelectKBest(f_classif)), 
                            ('classifier', classifier)])

    p_value_pipe_params = deepcopy(classifier_params)

    p_value_pipe_params.update({'select__k':[10,15,20]})

    select_features(p_value_pipe, p_value_pipe_params)  

    # %% [markdown]
    # ## Wrappers

    # %%
    print_to_results("Wrappers")

    # %% [markdown]
    # ### Decision Tree

    # %%
    print_to_results("Decision Tree")

    tree_pipe = Pipeline([('select', SelectFromModel(DecisionTreeClassifier())), 
                        ('classifier', classifier)])

    tree_pipe_params = deepcopy(classifier_params)

    tree_pipe_params.update({
                'select__max_features': [20],
                'select__estimator__max_depth': [None, 1, 3, 5]
                })

    select_features(tree_pipe, tree_pipe_params)  


    # %% [markdown]
    # ### Logistic Regression

    # %%
    print_to_results("Logistic Regression")

    logistic_pipe = Pipeline([('select', SelectFromModel(LogisticRegression())), 
                            ('classifier', classifier)])

    logistic_pipe_params = deepcopy(classifier_params)

    logistic_pipe_params.update({
                'select__max_features': [20]
                })
                
    select_features(logistic_pipe, logistic_pipe_params) 

    # %% [markdown]
    # ### SVC

    # %%
    print_to_results("SVC")

    svc_pipe = Pipeline([('select', SelectFromModel(LinearSVC())), 
                        ('classifier', classifier)])

    svc_pipe_params = deepcopy(classifier_params)

    svc_pipe_params.update({
                    'select__max_features': [20],
                    'select__estimator__dual': [True, False]
                })

    select_features(svc_pipe, svc_pipe_params) 

    # %%
    print_to_results(f'Finishing Feature Selection at {datetime.now()}')

# %%
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=5)
    pool.map(execute_feature_selection, datasets)


