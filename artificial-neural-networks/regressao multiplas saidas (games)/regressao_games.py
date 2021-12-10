import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
base = pd.read_csv('games.csv')

base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0) #apaga todas as linhas que possuem "NaN"
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name #armazenando os nomes 
base = base.drop('Name', axis = 1) 

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

labelenconder = LabelEncoder()
previsores[:, 0] = labelenconder.fit_transform(previsores[:, 0]) #transforrmando em valores numericos
previsores[:, 2] = labelenconder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelenconder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelenconder.fit_transform(previsores[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
previsorees = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape = 61,)
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada) #fazendo uma sequencia entre as camadas, já que o sequenciaal não está em uso
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation = 'linear')(camadaa_saida1)
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_saida2)

regressor = Model(inputs = camada_entrada, outputs = [camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs = 5000, batch_size = 100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)