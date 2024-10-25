import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

base_init = pd.read_excel('DatasourceCasePlan.xlsx')

base_ajust0 = base_init.copy(deep=True)
base_ajust0.columns = [col if col != '' else '0' for col in base_ajust0.columns]
columns_inter = base_ajust0.columns
new_columns = []

for ind, cols in enumerate(columns_inter):
    if cols == '0' and ind > 0:
        new_columns.append(new_columns[-1])
    else:
        new_columns.append(cols)

base_ajust0.columns = new_columns

base_transp = base_ajust0.T.reset_index()
base_transp['index'] = base_transp['index'].apply(lambda x: 0 if 'Unnamed: ' in str(x) else x)
base_transp.rename(columns = base_transp.iloc[0].to_dict(), inplace = True)
base_transp.rename(columns = {0: 'Dia_0', 'Grupos': 'Hora'}, inplace = True)
base_transp.drop(base_transp.index[0], inplace = True)
base_transp= base_transp.reset_index(drop = True)
base_transp.drop(base_transp.index[0], axis =0, inplace = True)
base_transp = base_transp.dropna(subset=['Hora'])
base_transp['Dia_0'] = base_transp['Dia_0'].replace(0, pd.NA)
base_transp['Dia_0'] = pd.to_numeric(base_transp['Dia_0'], errors='coerce')
base_transp['Dia_0'] = base_transp['Dia_0'].ffill()
base_transp = base_transp.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
base_transp = base_transp.astype(int).reset_index()
base_transp['Dia'] = pd.to_datetime('2023-12-31') + pd.to_timedelta(base_transp['Dia_0'], unit='days')
base_transp = base_transp.drop('Dia_0', axis = 1)
base_transp['Abertura_H'] = 8
base_transp['Encerramento_H'] = 22
base_transp['Dif_hora_Abert'] = base_transp['Hora'] - base_transp['Abertura_H']
base_transp['Dif_hora_Encer'] = base_transp['Encerramento_H'] - base_transp['Hora']
base_transp['Solic_Dentro_Expediente'] = (base_transp['Hora'] <= base_transp['Encerramento_H']).astype(int)
base_transp = base_transp.loc[(base_transp['Dia'] != 31) | (base_transp['Hora'] <= 22)]
dias_unicos = sorted(base_transp['Dia'].unique())

for ind in base_transp.index:
    dia_atual = base_transp.at[ind, 'Dia']
    if base_transp.at[ind, 'Solic_Dentro_Expediente'] == 0:
        proximo_dia = min([dia for dia in dias_unicos if dia > dia_atual], default=dia_atual)
        base_transp.at[ind, 'Dia_Atend'] = proximo_dia
    else:
        base_transp.at[ind, 'Dia_Atend'] = dia_atual

base_transp.drop(['Dia'], axis = 1, inplace = True)
base_transp.rename(columns = {'Dia_Atend': 'Dia'}, inplace = True)
semana = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
semana_ord = {'Domingo': 1, 'Segunda': 2, 'Terça': 3, 'Quarta': 4, 'Quinta' : 5, 'Sexta': 6, 'Sábado': 7}

base_transp['Semana'] = base_transp['Dia'].dt.day_name().map(semana)
base_transp = base_transp[['Semana', 'Dia', 'Hora', 'Grupo 1', 'Grupo 2', 'Grupo 3','Grupo 4', 'Grupo 5', 'Grupo 6', 'Total Geral']]

base_transp = base_transp.sort_values(by = ['Dia', 'Hora'], ascending = True)
base_transp = base_transp.reset_index(drop = True)
base = base_transp.copy(deep=True)

# Q3
distr_soli_por_grupo_Hora0 = pd.melt(base,
                               id_vars = ['Dia','Hora'],
                               var_name='Grupo', value_name='Solicitacoes')
distr_soli_por_grupo_Hora0['Solicitacoes'] = pd.to_numeric(distr_soli_por_grupo_Hora0['Solicitacoes'], errors='coerce')
distr_soli_por_grupo_Hora1 = distr_soli_por_grupo_Hora0.loc[distr_soli_por_grupo_Hora0['Grupo'].values != 'Total Geral'].copy(deep=True)
distr_soli_por_grupo_Hora1['Dia'] = distr_soli_por_grupo_Hora1['Dia'].dt.day.astype(int)
pivot = distr_soli_por_grupo_Hora1.pivot_table(values='Solicitacoes', index='Dia', columns='Hora', aggfunc='sum', fill_value=0)
distr_soli_por_grupo_Hora2 = distr_soli_por_grupo_Hora0.groupby(['Hora', 'Grupo']).agg({'Solicitacoes':'sum'}).reset_index()
distr_soli_por_grupo_Hora2 = distr_soli_por_grupo_Hora2.loc[distr_soli_por_grupo_Hora2['Grupo'] != 'Semana']

# Q4
distr_soli_por_grupo_Hora0_1 = pd.melt(base,
                               id_vars = ['Dia','Hora','Semana'],
                               var_name='Grupo', value_name='Solicitacoes')
distr_soli_por_grupo_Hora0_1['Solicitacoes'] = pd.to_numeric(distr_soli_por_grupo_Hora0_1['Solicitacoes'], errors='coerce')
distr_soli_por_grupo_Hora11 = distr_soli_por_grupo_Hora0_1.loc[distr_soli_por_grupo_Hora0_1['Grupo'].values == 'Total Geral'].copy(deep=True)
dis_solic_semana_0 = distr_soli_por_grupo_Hora11.groupby(['Semana', 'Hora']).agg({'Solicitacoes': 'sum'}).reset_index()
dis_solic_semana = dis_solic_semana_0.sort_values(by = 'Semana', key = lambda x: x.map(semana_ord), ascending=True)
###################################
# Model
base_nn = base.groupby(['Dia'], as_index = False).agg({'Total Geral': 'sum'})
X = base_nn[['Total Geral']]#[:-10] # 19 dados
y = base_nn[['Total Geral']]#[len(X):] # 10 dados

# X = base_nn[['Total Geral']][:-10]
# y = base_nn[['Total Geral']][:-10]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=123, shuffle = False)

data_train = torch.tensor(X_scaled, dtype=torch.float32)
data_pred = torch.tensor(y_scaled, dtype=torch.float32)

# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class model_NN_LSTM(nn.Module):
    def __init__(self):
        super(model_NN_LSTM, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=15, num_layers=1, batch_first=True)
        self.linear = nn.Linear(15, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

modelo = model_NN_LSTM()
criterio = nn.MSELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

epochs = 5000
for epoch in range(epochs):
    modelo.train()
    otimizador.zero_grad()
    y_pred = modelo(data_train)
    loss = criterio(y_pred, data_train)
    loss.backward()
    otimizador.step()

future_pred = []
num_dias = 5
aprendizado_historico = len(y_scaled)
ult_valor_conh = data_train[-aprendizado_historico:].detach().numpy().reshape(1, aprendizado_historico, -1)

modelo.eval()
with torch.no_grad():
    predict = modelo(data_pred)
    test_loss = criterio(predict, data_pred)
    mse = mean_squared_error(data_pred.numpy(), predict.numpy())
    print(f'Test Loss: {test_loss.item():.6f}, MSE: {mse:.4f}')

    for _ in range(num_dias):
      pred_fut = modelo(torch.tensor(ult_valor_conh, dtype = torch.float32))
      pred_value = pred_fut.numpy().flatten()[-aprendizado_historico]
      future_pred.append(pred_value.item())
      ult_valor_conh = np.append(ult_valor_conh[:,1:, :], [[[pred_value]]], axis=1)

pred_inv = scaler.inverse_transform(predict.numpy())
test_inv = scaler.inverse_transform(data_pred.numpy())

dias = base_nn['Dia'].reset_index(drop=True)
# dias = base_nn[len(data_train) : int(len(base_nn)/2)]['Dia'].reset_index(drop=True)
Real_data= pd.Series(test_inv[:].flatten(), name='Reais').reset_index(drop=True)
Pred_data = pd.Series(pred_inv.flatten(), name='Previstos').reset_index(drop=True)
df_Real_Pred = pd.DataFrame({'Dia': dias, 'Reais': Real_data, 'Previstos': Pred_data})

ultimo_dia = df_Real_Pred['Dia'].max() + pd.to_timedelta(1, unit='D')
pred_fut_inv = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))
fut_dia = [ultimo_dia + pd.DateOffset(days=i) for i in range(0, num_dias)]
future = pd.DataFrame({'Dia': fut_dia, 'Previstos': pred_fut_inv.flatten()})
df_general = pd.concat([df_Real_Pred[['Dia', 'Reais', 'Previstos']],future], ignore_index=True)

df_general_1 = df_general.copy(deep=True)
df_general_1['Semana'] = df_general_1['Dia'].dt.day_name().map(semana)
df_general_1 = df_general_1.loc[df_general_1['Dia'] <= '2024-01-31']
df_predit01 = df_general_1.groupby(['Semana'], as_index = False).agg({'Previstos': 'sum', 'Reais': 'sum'})
df_predit01 = df_predit01.sort_values(by = 'Semana', key = lambda x: x.map(semana_ord), ascending=True)
base_model_2 = df_predit01.copy(deep=True)

df_general_2 = df_general.copy(deep=True)
df_general_2['Semana'] = df_general_2['Dia'].dt.day_name().map(semana)
df_general_2 = df_general_2.loc[df_general_2['Dia'] > '2024-01-31']
df_predit02 = df_general_2.groupby(['Semana'], as_index = False).agg({'Previstos': 'sum', 'Reais': 'sum'})
df_predit02 = df_predit02.sort_values(by = 'Semana', key = lambda x: x.map(semana_ord), ascending=False)
# base_model_3 = df_predit02.copy(deep=True)

# Questão 1
with open ('base.pkl', 'wb') as data:
    pickle.dump(base, data)

# Questão 3
with open ('base_Q3_1.pkl', 'wb') as data:
    pickle.dump(pivot, data)

with open ('base_Q3_2.pkl', 'wb') as data:
    pickle.dump(distr_soli_por_grupo_Hora2, data)

# Questão 4
with open ('base_Q4_1.pkl', 'wb') as data:
    pickle.dump(dis_solic_semana, data)

# Modelo
with open ('base_model_1.pkl', 'wb') as data:
    pickle.dump(df_general, data)

with open ('base_model_2.pkl', 'wb') as data:
    pickle.dump(df_predit01, data)

with open ('base_model_3.pkl', 'wb') as data:
    pickle.dump(df_predit02, data)