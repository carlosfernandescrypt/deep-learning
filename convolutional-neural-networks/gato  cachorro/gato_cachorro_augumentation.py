from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
#[Conv2D]o recomendado é começar com 64 unidadades(onde tem 32), mass  pra  não dificultar o processamento, colocamos apenas 
#32, o numero 3 no final significa que iremos trabalhar com imagens rgb, ou imagens coloridas

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu')) #a função relu como explicado, serve pra tirar os valores negativos ou partes escuras das imagens
classificador.add(BatchNormalization()) # essa função vai acelerar o processamento, vai pegar o mapa de caracteriscas que foi gerado pela multiplicação pelo kernel, e vai transformar em valores numa escala entre 0 e 1
classificador.add(MaxPooling2D(pool_size = (2,2))) #pegar o maior valor, como visto nas aulas teoricas

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten()) #vai passar de uma matriz para um vetor para que assim possa ser colocado na rede neural

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))#camda de saida, por se tratar de uma rede que deve retorna gato ou cachorro, usaremos a sigmoid(classificação binaria)

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento  = ImageDataGenerator(rescale = 1./255, rotation_range = 7,  horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07, zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)
base_treinamento = gerador_treinamento.flow_from_directory('/content/drive/My Drive/dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('/content/drive/My Drive/dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000, epochs = 5, validation_data = base_teste, validation_steps = 1000)
