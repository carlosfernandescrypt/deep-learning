import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,  Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))       #esses dados são com base no tuning, lá ele deu os melhores parametros (necessario spyder)
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

novo = np.array([[15.80, 8.34, 112, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                  0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

previsao = classificador.predict(novo) #no spyder o valor daria 1, que no caso é maligno
previsao = (previsao > 0.5) #retorna true ou false

#print(previsao)
