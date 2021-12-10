import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap = 'gray') #transforma a imagem em cinza
plt.title('Classe' + str(y_treinamento[0])) #mostrar a classe da imagem

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1) #altura x largura e quantidade de canais, por se tratar apenas de cinza, é so 1 canal
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32') #transformando os valores em float, para que possa fazer a divisão por 225, e ficar com valores entre 1 e 0
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10) #por estar trabalhando com classificação de mais de 2 classes, usaremos a função softmax
classe_teste = np_utils.to_categorical(y_teste,10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = 'relu')) #operador convolucional (3,3 = o dectector de caracteres)
classificador.add(BatchNormalization()) #coloca os valores do mapa de caracteres entre 0 e 1, isso diminui o temmpo de carregamento
classificador.add(MaxPooling2D(pool_size = (2,2))) #tamanho do quadrado

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())


classificador.add(Dense(units = 128, activation = 'relu')) #camada de entrada
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax')) #camada de saída 
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 128, epochs = 5, validation_data = (previsores_teste, classe_teste))


resultado = classificador.evaluate(previsores_teste, classe_teste)
