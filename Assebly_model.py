import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform

from mlens.ensemble import SuperLearner
from sklearn.ensemble import VotingClassifier

from joblib import dump, load

SEED = 19
np.random.seed(SEED)

def getModels():
    
    params = {'C': 0.3712032345629517, 'penalty': 'l2'}
    model_1 = LogisticRegression(**params)

    params = {'n_neighbors': 17}
    model_2 = KNeighborsClassifier(**params)

    params = {'C': 0.1, 'kernel': 'linear'}
    model_3 = SVC(**params)

    params = {'criterion': 'entropy', 'max_depth': 3, 'max_features': 3, 'min_samples_leaf': 2}  
    model_4 = DecisionTreeClassifier(**params)

    params = {'learning_rate': 0.01, 'n_estimators': 50}
    model_5 = AdaBoostClassifier(**params)

    params = {'learning_rate': 0.01, 'n_estimators': 100}
    model_6 = GradientBoostingClassifier(**params)

    model_7 = GaussianNB()
    model_8 = RandomForestClassifier()
    model_9 = ExtraTreesClassifier()

    # create the sub models
    estimators = [('LR',model_1), ('KNN',model_2), ('SVC',model_3),
                ('DT',model_4), ('ADA',model_5), ('GB',model_6),
                ('NB',model_7), ('RF',model_8),  ('ET',model_9)]
    
    return estimators

# Загрузка данных
df = pd.read_csv('input/diabetes.csv')
# Получение имен колонок
df_name=df.columns

#Разделение данных
X = df[df_name[0:10]]
Y = df[df_name[10]]
X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                test_size=1/3,
                                                random_state=SEED,
                                                stratify=df['Outcome'])


#Обучение и получение результатов на обучающей выборке
kfold = StratifiedKFold(n_splits=15, random_state=SEED)
ensemble = VotingClassifier(getModels())
results = cross_val_score(ensemble, X_train, y_train, cv=kfold)
print('Accuracy on train: ',results.mean())

#Обучение модели
ensemble_model = ensemble.fit(X_train,y_train)

#Сохранение обученной модели в файл
dump(ensemble_model, 'ml_model.joblib') 

#Проверка модели на всем наборе данных
pred = ensemble_model.predict(X)
print('Accuracy on test:' , (Y == pred).mean())
