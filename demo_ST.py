#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:33:31 2022

@author: lise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STREAMLIT
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<h1 style="color:#4500ef;">SÉRIES TEMPORELLES 🇦🇺🐨</h1>', unsafe_allow_html=True)

st.markdown("***")

st.markdown("Prédictions des températures maximales sur l'année 2021 + 6 premiers mois 2022.")
st.markdown("Données récupérées sur les températures maximales autour de Melbourne entre 1995 et 2021. Moyennes par mois par année.")



# LECTURE DES FICHIERS ET "MISE EN FORME"

debut = pd.read_csv("mean_max_Temp_1919-1975.csv")
milieu = pd.read_csv("mean_max_Temp_1953-2006.csv")
fin = pd.read_csv("mean_max_Temp_1995-2021.csv")

debut.drop(['Product code', 'Station Number', 'Annual'], axis = 1, inplace = True)
milieu.drop(['Product code', 'Station Number', 'Annual'], axis = 1, inplace = True)
fin.drop(['Product code', 'Station Number', 'Annual'], axis = 1, inplace = True)

# pour les trois df: montrer les lignes des df comprenant des NaN
slice_object = slice(2, 14)
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

debut_final = pd.DataFrame(columns = ['Date', 'Mean_Max_Temp'])

for row in debut.itertuples():
    for temp, month in zip(row[slice_object], months):
        date = str(row.Year) + '-' + str(month)
        debut_final = debut_final.append({'Date' : date , 'Mean_Max_Temp' : temp} , ignore_index=True)
        

def transform_df(df):
    slice_object = slice(2, 14)
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    new_df = pd.DataFrame(columns = ['Date', 'Mean_Max_Temp'])

    for row in df.itertuples():
        for temp, month in zip(row[slice_object], months):
            date = str(row.Year) + '-' + str(month)
            new_df = new_df.append({'Date' : date , 'Mean_Max_Temp' : temp} , ignore_index=True)
            
    return new_df

series_debut = transform_df(debut)
series_milieu = transform_df(milieu)
series_fin = transform_df(fin)

# DEBUT: Si certaines dates se chevauchent dans les trois df et que l'un d'elles est NaN et pas les autres, remplacer les NaN par les autres
for row in series_debut.itertuples():
    if pd.isna(row.Mean_Max_Temp):
        other = series_milieu[series_milieu['Date'] == row.Date]['Mean_Max_Temp'].values
        if len(other) > 0:
            series_debut.at[row.Index, 'Mean_Max_Temp'] = other[0]
            # series_debut.loc[row.Index]['Mean_Max_Temp'] = other[0]
   

# IDEM POUR MILIEU
for row in series_milieu.itertuples():
    if pd.isna(row.Mean_Max_Temp):
        
        other_debut = series_debut[series_debut['Date'] == row.Date]['Mean_Max_Temp'].values
        other_fin = series_fin[series_fin['Date'] == row.Date]['Mean_Max_Temp'].values
        
        if len(other_debut) > 0:
            series_milieu.at[row.Index, 'Mean_Max_Temp'] = other_debut[0]

        if len(other_fin) > 0:
            series_milieu.at[row.Index, 'Mean_Max_Temp'] = other_fin[0]


# FIN
for row in series_fin.itertuples():
    if pd.isna(row.Mean_Max_Temp):
        other = series_milieu[series_milieu['Date'] == row.Date]['Mean_Max_Temp'].values
        if len(other) > 0:
            series_fin.at[row.Index, 'Mean_Max_Temp'] = other[0]
            






# ST
# PRÉPARATION DES DONNÉES 
series_fin.dropna(inplace = True)
series_fin.set_index('Date', inplace = True)
series_fin = series_fin.squeeze()

data = st.button("Jeter un coup d'oeil aux données")

if data:
    st.table(series_fin[:6])
    

st.markdown("Graphique de l'évolution des températures maximales aux alentours de Melbourne, sur 26 ans:")


fig = plt.figure(figsize = (35, 25))

plt.plot(series_fin, color = 'c')

# Réarrangement des labels de xticks
ticks = np.arange(0, len(series_fin), 12)
#series_fin.index

x_labels = []

for x in ticks:
    x_labels.append(series_fin.index[x].split("-")[0])

plt.xticks(ticks, x_labels, rotation=45)
plt.xticks(size = 20)
plt.yticks(size = 20)

# Title
plt.title('Températures maximales moyennes, par mois, sur Melbourne. Evolution sur 26 ans',  fontsize = 25);

st.pyplot(fig)

st.markdown("*Il semble bien y avoir un phénomène cyclique des températures maximales sur une année, avec un pic (valeurs moyennes mensuelles) en début d'année (été) et une chute en milieu d'année (hiver). Cela se reproduit bien tous les ans.*")




# DECOMPOSITION SAISONNIERE
st.markdown('<h6 style="color:#9dacf6;"><u>Décomposition saisonnière & Différenciations</u></h6>', unsafe_allow_html=True)


from statsmodels.tsa.seasonal import seasonal_decompose

st.markdown("Décomposition saisonnière:")
res = seasonal_decompose(series_fin, period = 12, model = 'a') # forcé avec period = 1, 12???
res.plot()
fig1 = plt.show()

st.pyplot(fig1)


#Différenciation: Eliminer les pics de saisonnalité

st.markdown("Stationnarisation par différenciations (1 différenciation + 1 différenciation saisonnière):")
series_fin1 = series_fin.diff().dropna()
pd.plotting.autocorrelation_plot(series_fin1);

series_fin2 = series_fin1.diff(periods = 12).dropna()
fig2 = pd.plotting.autocorrelation_plot(series_fin2).plot()

st.pyplot(fig2)


# Test stat:
from statsmodels.tsa.stattools import adfuller
#adfuller(series_fin2)

_, p_value, _, _, _, _  = adfuller(series_fin2)

st.write("p-value:", p_value)


# ACF et PACF 
st.markdown('<h6 style="color:#9dacf6;"><u>Autocorrélation et Autocorrélation partielle</u></h6>', unsafe_allow_html=True)

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

fig3 = plt.figure(figsize= (14,7))

plt.subplot(121)
plot_acf(series_fin2, lags = 36, ax=plt.gca())

plt.subplot(122)
plot_pacf(series_fin2, lags = 36, ax=plt.gca())

plt.show()

st.pyplot(fig3)




# APPLICATION DU MODELE
st.markdown('<h6 style="color:#9dacf6;"><u>Application du modèle SARIMAX</u></h6>', unsafe_allow_html=True)

import statsmodels.api as sm
import warnings
warnings.simplefilter('always', category=UserWarning)

model= sm.tsa.SARIMAX(series_fin, order=(1,1,1),seasonal_order=(0,1,1,12))
sarima=model.fit()

st.write(sarima.summary())



# PREDICTIONS
st.markdown('<h6 style="color:#9dacf6;"><u>Prédictions & Visualisations</u></h6>', unsafe_allow_html=True)

st.markdown("Prédictions pour l'année 2021:")
pred = sarima.predict(312, 321) #Prédiction sur la dernière année = 2021 (Janvier - Octobre)

series = series_fin.to_frame()

# Changer les index de pred sinon pb avec TimeStamp, format différent de la série de valeurs réelles. 
new_index = []

for i in pred.index:
    new_index.append(i.strftime("%Y-%m"))


pred.index = new_index


fig4 = plt.figure(figsize = (15, 10))

plt.plot(series_fin, color = 'c', label = "Valeurs réelles") #Visualisation
plt.plot(pred, '--r', label = "Prédictions")
plt.legend()

plt.axvline(x= pred.index[0], color='r'); # Ajout de la ligne verticale

plt.xticks(ticks, x_labels, rotation=45);

st.pyplot(fig4)


futur1 = st.button("Tableau des prédictions versus valeurs réelles sur 2021")
if futur1:
    final = series_fin[-10:].to_frame('Reel').join(pred.to_frame('Prédit'))
    st.table(data = final)



st.markdown("Prédictions pour la fin de l'année 2021 / Début d'année 2022:")

pred_futur = sarima.predict(321, 329)

fut_index = []

for i in pred_futur.index:
    fut_index.append(i.strftime("%Y-%m"))

pred_futur.index = fut_index

fig5 = plt.figure(figsize = (15, 10))

plt.plot(series_fin, color = 'c', label = "Valeurs réelles") #Visualisation
plt.plot(pred, '--r', label = "Prédictions deux premiers trimestres 2021")
plt.plot(pred_futur, '--g', label = "Prédictions fin d'année 2021 + début 2022")
plt.legend()

plt.axvline(x= pred.index[0], color='r'); # Ajout de la ligne verticale

plt.xticks(ticks, x_labels, rotation=45);

st.pyplot(fig5)


futur2 = st.button("Tableau des prédictions versus valeurs réelles sur fin 2021/ début 2022")
if futur2:
    st.table(data= pred_futur)


