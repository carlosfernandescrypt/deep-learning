import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values 
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(optimizer, loss, activation, neurons):
  classificador = Sequential()
  classificador.add(Dense(units = neurons, activation = activation, input_dim = 4))
  classificador.add(Dense(units = neurons, activation = activation))
  classificador.add(Dense(units = 3, activation = 'softmax'))
  classificador.compile(optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
  return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30], 'epochs': [50,100,], 'optimizer': ['adam', 'sgd'], 'loss': ['categorical_crossentropy', 'hinge'],
              'activation': ['relu', 'tanh'], 'neurons': [4, 8]}

grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score

print(melhores_parametros)
print(melhor_precisao)
