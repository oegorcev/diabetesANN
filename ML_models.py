import sys
import os
import pandas as pd
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform

SEED = 42
np.random.seed(SEED)

#Стандартные модели
def getDefaultModels():

    defaultModel = []
    defaultModel.append(('LR'   , LogisticRegression()))
    defaultModel.append(('LDA'  , LinearDiscriminantAnalysis()))
    defaultModel.append(('KNN'  , KNeighborsClassifier()))
    defaultModel.append(('CART' , DecisionTreeClassifier()))
    defaultModel.append(('NB'   , GaussianNB()))
    defaultModel.append(('SVM'  , SVC(probability=True)))
    defaultModel.append(('AB'   , AdaBoostClassifier()))
    defaultModel.append(('GBM'  , GradientBoostingClassifier()))
    
    return defaultModel

#Модели с применение скалирования
def getScaledModels(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()

    models = []
    models.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))
    models.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    models.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    models.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    models.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    models.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    models.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    models.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    return models 

#Обучение и получение результатов
def learnModels(X_train, y_train,models):
    # Параметры для обучения
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
    return names, results

def TurkyFences(df,nameOfFeature,drop=False):
    
    #Количество записей
    valueOfFeature = df[nameOfFeature]

    # Вычисление первого квартиля Q1 (25 процентов данных для заданного параметра)
    Q1 = np.percentile(valueOfFeature, 25.)

    # Вычисление квартиля Q3 (75 процентов данных для заданного параметра)
    Q3 = np.percentile(valueOfFeature, 75.)

    # Нахождения шага
    step = (Q3 - Q1) * 1.5

    # Вычисление outliers
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values

    # Удаление outliers
    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers), feature_outliers))
    if drop:
        good_data = df.drop(df.index[outliers]).reset_index(drop = True)
        print ("New dataset with removed outliers has {} samples with {} features each. feature: ".format(*good_data.shape) + nameOfFeature)
        return good_data
    else: 
        print ("Nothing happens, df.shape = ",df_out.shape)
        return df_out

def makeResultScores(names,results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model ':names, 'Score': scores})
    return scoreDataFrame

# Загрузка данных
df = pd.read_csv('input/diabetes.csv')
# Получение имен колонок
df_name=df.columns

#Разделение данных на вход и выход
X = df[df_name[0:10]]
Y = df[df_name[10]]

#Обучение с помощью стандартных моделей
models = getDefaultModels()
names,results = learnModels(X, Y, models)
basedLineScore = makeResultScores(names,results)

#Обучение с применением стандартного скалирования
models = getScaledModels('standard')
names,results = learnModels(X, Y, models)
scaledScoreStandard = makeResultScores(names,results)

#Обучение с применением МинМакс скалирования
models = getScaledModels('minmax')
names,results = learnModels(X, Y,models)
scaledScoreMinMax = makeResultScores(names,results)

#Применение правила Тьюки
for i in range(11):
    if i == 0:
        df_clean = TurkyFences(df,df_name[i],True)
    if i == 1:
        df_clean = df_clean
    else:
        df_clean = TurkyFences(df_clean,df_name[i],True)

#Разделение "очищенных" данных после применения правила Тьюки
df_clean_name = df_clean.columns
X_c = df_clean[df_clean_name[0:10]]
Y_c = df_clean[df_clean_name[10]]

#Обучение на очищенных данных с применением МинМакс скалирования
models = getScaledModels('minmax')
names,results = learnModels(X_c, Y_c,models)
scaledScoreMinMax_c = makeResultScores(names,results)

#Выделение из данных наиболее важных параметров
df_feature_imp=df[['Age','Gender', 'Exercise','Education', 'HBP','FVC', 'DK','Outcome']]
df_feature_imp_name = df_feature_imp.columns
X = df_feature_imp[df_feature_imp_name[0:df_feature_imp.shape[1]-1]]
Y = df_feature_imp[df_feature_imp_name[df_feature_imp.shape[1]-1]]

#Обучение на данных с важными параметрами с применением МинМакс скалирования
models = getScaledModels('minmax')
names,results = learnModels(X, Y,models)
scaledScoreMinMax_im = makeResultScores(names,results)

#Построение итоговой таблицы для сравнения точностей
#########################################################################################
compareModels = pd.concat([ basedLineScore,
                            scaledScoreStandard,
                            scaledScoreMinMax,
                            scaledScoreMinMax_c,
                            scaledScoreMinMax_im], axis=1)

print(compareModels)
    
