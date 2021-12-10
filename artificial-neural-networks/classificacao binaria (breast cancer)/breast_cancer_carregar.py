import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

novo = np.array([[15.80, 8.34, 112, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                  0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

previsao = classificador.predict(novo) #no spyder o valor daria 1, que no caso é maligno
previsao = (previsao > 0.5) #retorna true ou false

#print(previsao)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['binaray_accuracy'])
resultado - classificador.evaluate(previsores, classe)

#print(resultado)