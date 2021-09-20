from fastai.tabular import *

if __name__ == '__main__':
 
    # Загрузка данных
    df = pd.read_csv('input/diabetes.csv')

    # Модификаторы нейросети
    procs = [FillMissing, Categorify]

    # Группировка параметров
    dep_var = 'Outcome'
    cat_names = ['Gender', 'Race', 'Education', 'Smoking', 'HBP', 'HD','Exercise','FVC','DK']
    cont_names = ['Age']
   
    # Подготовка данныз к обучению
    data = (TabularList
                    .from_df(df=df, cat_names=cat_names, cont_names=cont_names, procs=procs)
                    .split_by_idx(list(range(800,1000)))
                    .label_from_df(cols=dep_var)
                    .databunch())

    
    # Обучение сети
    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
    learn.fit_one_cycle(5, 1e-2)

    # Сохранение модели в файл
    learn.export('fastai_model.pkl')