import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
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
classificador.add(MaxPooling2D(pool_size = (2,2))) #tamanho do quadrado
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu')) #camada de entrada
classificador.add(Dense(units = 10, activation = 'softmax')) #camada de saída 
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

################### nova codificação para gerar novas imagens

gerador_treinamento = ImageDataGenerator(rotation_range = 7, horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07, zoom_range = 0.2)
gerador_teste = ImageDataGenerator()
base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size = 128)

base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size = 128)
classificador.fit_generator(base_treinamento, steps_per_epoch = 60000 / 128, epochs = 5, validation_data = base_teste, validation_steps = 10000 / 128)