import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

SEED = 42
np.random.seed(SEED)

# Загруза и обработка данных
dataset = pd.read_csv('input/diabetes.csv')
features = list(dataset.columns.values)
features.remove('Outcome')
X_Input = (dataset[features]).values

# Скалирование
scaler = MinMaxScaler()
scaler.fit(X_Input)

# Разделение данных
X = scaler.transform(X_Input)
Y = dataset['Outcome'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)


# Описание модели
model = Sequential()

model.add(Dense(256, input_dim=10, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.summary()
################################################################################################

# Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
fit_machine = model.fit(x_train, 
                      y_train,
                      epochs=40,
                      batch_size=32,
                      validation_split=0.10,
                      verbose=1)

# Сохранение модели в файл
model.save('keras_model.h5')

# Выгрузка модели из файла
#model = tf.keras.models.load_model('my_model.h5')

# Оценивание результатов
result = model.evaluate(x_test, y_test, verbose=0)
print(result)
print('accuracy: ', result[1]*100)