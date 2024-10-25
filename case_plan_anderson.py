import streamlit as st
import pickle
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

base = pickle.load(open('base.pkl', 'rb'))
base_Q3_1 = pickle.load(open('base_Q3_1.pkl', 'rb'))
base_Q3_2 = pickle.load(open('base_Q3_2.pkl', 'rb'))
base_Q4_1 = pickle.load(open('base_Q4_1.pkl', 'rb'))
base_model_1 = pickle.load(open('base_model_1.pkl', 'rb'))
base_model_2 = pickle.load(open('base_model_2.pkl', 'rb'))
base_model_3 = pickle.load(open('base_model_3.pkl', 'rb'))

# Dados Complementares:
Total_Solicitacoes_mes = sum(base['Total Geral'])
dias_funcionamento = 31-8 # Considerando o dia 1 na segunda-feira
absenteismo_mes = 0.03
TurnOver_mes = 0.020
fator_de_reducao = (1 / 1 - absenteismo_mes - TurnOver_mes)
canal_email = 0.38
canal_chat = 0.32
canal_whatsapp = 0.30
produtividade_email_indiv_dia = 25
produtividade_chat_indiv_dia = 35
produtividade_whatsapp_indiv_dia = 35
hrs_funcionamento_horas = 14
funcionamento_diario_dias = 5
jornada_diaria_horas = 8
custo_por_funcionario = 6000
melhoria_eficiencia = 0.20

# Q1
solicitacoes_email = Total_Solicitacoes_mes * canal_email
solicitacoes_chat = Total_Solicitacoes_mes * canal_chat
solicitacoes_whatsapp = Total_Solicitacoes_mes * canal_whatsapp
solicitacoes_email_dia = solicitacoes_email / dias_funcionamento
solicitacoes_chat_dia = solicitacoes_chat / dias_funcionamento
solicitacoes_whatsapp_dia = solicitacoes_whatsapp / dias_funcionamento
pessoas_email = np.ceil(solicitacoes_email_dia / produtividade_email_indiv_dia)
pessoas_chat = np.ceil(solicitacoes_chat_dia / produtividade_chat_indiv_dia)
pessoas_whatsapp = np.ceil(solicitacoes_whatsapp_dia / produtividade_whatsapp_indiv_dia)
pessoas_email_ajustado = int(np.ceil(pessoas_email * fator_de_reducao))
pessoas_chat_ajustado = int(np.ceil(pessoas_chat * fator_de_reducao))
pessoas_whatsapp_ajustado = int(np.ceil(pessoas_whatsapp * fator_de_reducao))

canais = ['E-mail', 'Chat', 'WhatsApp']
pessoas = [pessoas_email_ajustado, pessoas_chat_ajustado, pessoas_whatsapp_ajustado]
Q1_qtd_pessoas_canal_mes = pd.DataFrame({'Canal': canais, 'Pessoas': pessoas})
grafico_Q1_1 = px.histogram(Q1_qtd_pessoas_canal_mes, x='Canal', y='Pessoas', color='Canal', text_auto = True, width=750, height=400, title= 'Comercial: Quantidade de Colaboradores')

# Q2
custo_atual_email = pessoas_email_ajustado * custo_por_funcionario
custo_atual_chat = pessoas_chat_ajustado * custo_por_funcionario
custo_atual_whatsapp = pessoas_whatsapp_ajustado * custo_por_funcionario
custo_total_atual = custo_atual_email + custo_atual_chat + custo_atual_whatsapp
pessoas_email_eficiencia = int(pessoas_email_ajustado * (1 - melhoria_eficiencia))
pessoas_chat_eficiencia = int(pessoas_chat_ajustado * (1 - melhoria_eficiencia))
pessoas_whatsapp_eficiencia = int(pessoas_whatsapp_ajustado * (1 - melhoria_eficiencia))
custo_novo_email = pessoas_email_eficiencia * custo_por_funcionario
custo_novo_chat = pessoas_chat_eficiencia * custo_por_funcionario
custo_novo_whatsapp = pessoas_whatsapp_eficiencia * custo_por_funcionario
custo_total_novo = custo_novo_email + custo_novo_chat + custo_novo_whatsapp

custo_atual = [custo_atual_email, custo_atual_chat, custo_atual_whatsapp]
custo_novo = [custo_novo_email, custo_novo_chat, custo_novo_whatsapp]
Q2_efic_custo_canal_mes_0 = pd.DataFrame({'Canal': canais, 'Custo ATUAL': custo_atual, 'Custo NOVO': custo_novo})
Q2_efic_custo_canal_mes = pd.melt(Q2_efic_custo_canal_mes_0, id_vars = 'Canal', value_vars = ['Custo ATUAL', 'Custo NOVO'], var_name = 'Tipo', value_name='Custo')
grafico_Q2_1 = px.histogram(Q2_efic_custo_canal_mes, y = 'Custo', labels = {'Canal': 'Canal de Atendimento'},
                        x = 'Canal', color ='Tipo', barmode = 'group', text_auto = True, width=750, height=400, title= 'Comparação de Custos por Canal')

# Q3
class graficos:
    def __init__(self, base_Q3_1, base_Q3_2):
        self.base_Q3_1 = base_Q3_1
        self.base_Q3_2 = base_Q3_2

    def grQ3_1(self):
        fig_line_1 = px.imshow(
            self.base_Q3_1,
            text_auto=True,
            height=700,
            width=1500,
            aspect='auto',
            title='Heatmap de Solicitações por Hora e Dia',
            labels=dict(x='Hora', y='Dia', color='Solicitações')
        )
        return fig_line_1

    def grQ3_2(self):
        fig_line_2 = px.line(
            self.base_Q3_2,
            x='Hora',
            y='Solicitacoes',
            color='Grupo',
            markers=True,
            title='Volume de Solicitações por Hora no Mês',
            height=500,
            width=1400
        )
        return fig_line_2

gr_class = graficos(base_Q3_1, base_Q3_2)
grafico_Q3_1 = gr_class.grQ3_1()
grafico_Q3_2 = gr_class.grQ3_2()

# Q4
semana_ord = {'Domingo': 1, 'Segunda': 2, 'Terça': 3, 'Quarta': 4, 'Quinta' : 5, 'Sexta': 6, 'Sábado': 7}
base_Q4_1 = base_Q4_1.sort_values(by = 'Semana', key = lambda x: x.map(semana_ord), ascending=False)
grafico_Q4_1 = px.scatter(base_Q4_1, x='Hora', y='Semana', title = 'Volume de Solicitações por Semana por Hora',
                    color = 'Solicitacoes', size = 'Solicitacoes', orientation = 'v', height= 600, width= 800)

# Model
df_tempo = base_model_1.copy(deep=True)
df_tempo['Ord_Semana'] = base_model_1['Dia'].dt.isocalendar().week


ultima_period_col_real = df_tempo['Dia'][df_tempo['Reais'].notna()].max()
def graphic(dataframe, period_col, real_col, predict_col):
    fig_model = go.Figure()
    fig_model.add_trace(go.Bar(
        x=dataframe[period_col],
        y=dataframe[real_col],
        name= real_col,
        marker_color='blue',
        opacity=0.7,
        text=round(dataframe[real_col], 1),
        textposition='auto'
    ))

    fig_model.add_trace(go.Bar(
        x=dataframe[period_col][dataframe[period_col] <= ultima_period_col_real],
        y=dataframe[predict_col][dataframe[period_col] <= ultima_period_col_real],
        name='Previstos (Período Real)',
        marker_color='brown',
        opacity=0.5,
        text= round(dataframe[predict_col][dataframe[period_col] <= ultima_period_col_real],1),
        textposition='auto'
    ))

    fig_model.add_trace(go.Bar(
        x=dataframe[period_col][dataframe[period_col] > ultima_period_col_real],
        y=dataframe[predict_col][dataframe[period_col] > ultima_period_col_real],
        name='Previstos (Período Futuro)',
        marker_color='black',
        opacity=0.5,
        text=round(dataframe[predict_col][dataframe[period_col] > ultima_period_col_real], 1),
        textposition='auto'
    ))

    fig_model.add_shape(
        type="line",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color="black", width=1.5),
        xref='paper',
        yref='y',
    )

    fig_model.update_layout(
        title='Comparativo de Reais e Previstos',
        xaxis_title=period_col,
        yaxis_title='Valores',
        barmode='group',  # Sobrepõe as barras para uma visualização comparativa
        template='plotly_white'
    )
    return fig_model

def graphic_semana(dataframe, period_col, real_col, predict_col, df_futuro):
    fig_model_sem = go.Figure()
    fig_model_sem.add_trace(go.Bar(
        x=dataframe[period_col],
        y=dataframe[real_col],
        name= real_col,
        marker_color='blue',
        opacity=0.7,
        text=round(dataframe[real_col], 1),
        textposition='auto'
    ))

    fig_model_sem.add_trace(go.Bar(
        x=dataframe[period_col],
        y=dataframe[predict_col],
        name='Previstos (Período Real)',
        marker_color='brown',
        opacity=0.5,
        text= round(dataframe[predict_col],1),
        textposition='auto'
    ))

    fig_model_sem.add_trace(go.Bar(
        x=df_futuro[period_col],
        y=df_futuro[predict_col],
        name='Previstos (Período Futuro)',
        marker_color='black',
        opacity=0.5,
        text=round(df_futuro[predict_col], 1),
        textposition='auto'
    ))

    fig_model_sem.add_shape(
        type="line",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color="black", width=1.5),
        xref='paper',
        yref='y',
    )

    fig_model_sem.update_layout(
        title='Comparativo de Reais e Previstos',
        xaxis_title=period_col,
        yaxis_title='Valores',
        barmode='group',  # Sobrepõe as barras para uma visualização comparativa
        template='plotly_white'
    )
    return fig_model_sem

grafico_model_2 = graphic_semana(base_model_2, 'Semana','Reais', 'Previstos', base_model_3)

# Dashboard
custo_atual_email = pessoas_email_ajustado * custo_por_funcionario
custo_atual_chat = pessoas_chat_ajustado * custo_por_funcionario
custo_atual_whatsapp = pessoas_whatsapp_ajustado * custo_por_funcionario
custo_total_atual = custo_atual_email + custo_atual_chat + custo_atual_whatsapp
pessoas_email_eficiencia = int(pessoas_email_ajustado * (1 - melhoria_eficiencia))
pessoas_chat_eficiencia = int(pessoas_chat_ajustado * (1 - melhoria_eficiencia))
pessoas_whatsapp_eficiencia = int(pessoas_whatsapp_ajustado * (1 - melhoria_eficiencia))
custo_novo_email = pessoas_email_eficiencia * custo_por_funcionario
custo_novo_chat = pessoas_chat_eficiencia * custo_por_funcionario
custo_novo_whatsapp = pessoas_whatsapp_eficiencia * custo_por_funcionario
custo_total_novo = custo_novo_email + custo_novo_chat + custo_novo_whatsapp

custo_atual = [custo_atual_email, custo_atual_chat, custo_atual_whatsapp]
custo_novo = [custo_novo_email, custo_novo_chat, custo_novo_whatsapp]
Q2_efic_custo_canal_mes_0 = pd.DataFrame({'Canal': canais, 'Custo ATUAL': custo_atual, 'Custo NOVO': custo_novo})
Q2_efic_custo_canal_mes = pd.melt(Q2_efic_custo_canal_mes_0, id_vars='Canal', value_vars=['Custo ATUAL', 'Custo NOVO'], 
                                  var_name='Tipo', value_name='Custo')

# Início do Streamlit
st.title('Case Planejamento - Contabilizei')
st.header('Anderson Sousa Pereira Sá')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Questão 1', 'Questão 2', 'Questão 3', 'Questão 4', 'Modelo Preditivo'])

# Questão 1
with tab1:
    st.header(' ⁂ 1 - Alocação de Pessoas por Canal:')
    st.write('Para melhor disponibilidade, performance e qualidade ao nosso cliente:')
    st.markdown(f''' 
    Considerando os dados fornecidos e objetivando o cumprimento da produtividade diária por colaborador, fornecendo \numa ótima entrega quanto a disponibilidade, performance e qualidade aos nossos clientes, seguem os detalhamentos \ndas quantidades de pessoas para cada Canal conforme detalhamento abaixo: \n 
    ✔ E-mail: {pessoas_email_ajustado}
    ✔ Chat: {pessoas_chat_ajustado} 
    ✔ WhatsApp: {pessoas_whatsapp_ajustado}
    ➤ Total do time: {pessoas_email_ajustado + pessoas_chat_ajustado + pessoas_whatsapp_ajustado}''')
    st.plotly_chart(grafico_Q1_1, use_container_width=True)

# Questão 2
with tab2:
    st.header('⁂ 2 - Propor ações para melhorarmos 20% de eficiência, considerando nossa performance o custo de hc em R$ 6.000,00')
    st.write(f'Custo total atual: R$ {custo_total_atual:,.2f}')
    st.markdown(f''' 
    De acordo com os dados, considerando uma melhoria de 20% em eficiência, foi obtivo as seguinites intepretações:\n 
    ✔ Custo atual total para o referido mês é: R$ {custo_total_atual:,.2f} \n 
    ✔ Custo com 20% de melhoria de eficiência: R$ {custo_total_novo:,.2f} \n \n
    Com isso proponho as segunites ações (sendo necessário verificar também a viabilidade): \n
    ➥ Treinar a equipe para melhorar no tempo de atendimento
    ➥ Implementação de chatbots para atender uma parcela das solicitações via whatsapp e chat
    ➥ Automatizar respostas padrão por e-mail\n
    ''')
    st.plotly_chart(grafico_Q2_1, use_container_width=True)

# Questão 3
with tab3:
    st.header('⁂ 3 - Quais são os principais insights que você levaria para o Diretor do time para melhorar tempo de resolução?')
    colQ3_1, colQ3_2 = st.columns(2)
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    with colQ3_2:
        st.markdown(f'''
            Conclusão e sugestões: \n
            ➥ Redistribuir a equipe para cobrir melhor os horários de pico. \n
            ➥ No dia 30 existe uma demanda ainda maior, é interessante ter uma atenção junto a equipe para mantermos a qualidade nos atendimentos. \n
            ➥ Com isso, é interessante a implementação dos chatsbots e templetes de e-mail para dar maior fluidez no atendimento.\n
                ''')
    with colQ3_1:

            sidebar_dash = st.radio("Escolha visão gráfica que deseja analisar", ['Horas por Dia', 'Horas por Mês'])
    
    st.columns(1)
    if sidebar_dash == 'Horas por Dia':
        st.plotly_chart(grafico_Q3_1, use_container_width=True)
    else:
        st.plotly_chart(grafico_Q3_2, use_container_width=True)

# Questão 4
with tab4:
    st.header('⁂ 4 - Quais insights podem ser gerados a respeito do comportamento dos nossos clientes a partir destas informações? Você teria alguma recomendação adicional?')
    st.markdown(f''' 
            Insights para tomada de decisões:\n
            ➥ Ajustar o horário de operação para cobrir melhor os horários de maior demanda.\n
            ➥ Considerar um atendimento reduzido em horários de pouca demanda para reduzir custos. \n
            ➥ Melhorar a integração dos canais para uma experiência mais fluida (por exemplo, transição de um chatbot para um atendente humano).
                ''')
    st.plotly_chart(grafico_Q4_1, use_container_width=True)

# Modelo Preditivo
with tab5:
    st.header('Modelo Preditivo de Solicitações de Clientes:')
    st.write('Dados importantes: Previsão de 10 dias para frente || Período de Aprendizagem: últimos 12 dias')
    col_M1, col_M2, col_M3 = st.columns(3)
    with col_M1:
        input_ord = st.number_input('Selecione abaixo entre a 1ª e 5ª semana do mês para visualizar no gráfico e 0 para o gráfico completo:', 
                              value = 0, min_value = 0, max_value= 5, step = 1, label_visibility = 'collapsed')
    
    if input_ord == 0:
      df_tempo = df_tempo
    else:
      df_tempo = df_tempo.loc[df_tempo['Ord_Semana'] == input_ord]
    st.plotly_chart(graphic(df_tempo,'Dia', 'Reais','Previstos'), use_container_width=True)
    st.plotly_chart(grafico_model_2, use_container_width=True)

# streamlit run case_plan_anderson.py