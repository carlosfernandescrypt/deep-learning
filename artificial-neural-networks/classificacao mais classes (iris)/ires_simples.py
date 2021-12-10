import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values #separa os previsores das classes das plantas
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)  #troca de classes (vai passar de string para numeros[0,1,2])
classe_dummy = np_utils.to_categorical(classe)

#iris setosa     1 0 0
#iris versicolor 0 1 0    nas camadas de saidas tem de ser assim (visualização melhor no spyder)
#iris virginica  0 0 1


previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))  #units = quantidades de entrada(4) + quantidade de saidas(3) / 2 = 3.5 (4) (input_dim = quantidade de previsores)
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax')) #a softmax é usada em problemas com mais de 2 classes senão é a sigmoid (2 classes)
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

#classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)


classe_teste2 = [np.argmax(t) for t in classe_teste]   #serve pra mostrar a quantidade de classes de cada flor
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)