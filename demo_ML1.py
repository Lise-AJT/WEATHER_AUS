#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:35:43 2022

@author: lise
"""

# IMPORTATION DES PACKAGES
import pandas as pd
import numpy as np

# VIZ
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

# ML: PRÉPARATION
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import RandomOverSampler

# ALGORITHME
from sklearn.ensemble import RandomForestClassifier

# ANALYSES
from sklearn import metrics

# STREAMLIT
import streamlit as st








## INTRODUCTION ##

st.markdown('<h1 style="color:#4500ef;">MACHINE LEARNING 🇦🇺🐨</h1>', unsafe_allow_html=True)

st.markdown("***")

st.markdown("Application d'un Random Forest après un cleaning plus approfondi et analyses des résultats.")










## CLEANING ##

st.markdown('<h2 style="color:#3adfb2;">1/ CLEANING</h2>', unsafe_allow_html=True)

# Importation du fichier
st.markdown("Les données sont importées et les valeurs manquantes sont étudiées.")

df = pd.read_csv("weatherAUS.csv")
pd.set_option('max_columns', None)

donnees = st.button("Jeter un coup d'oeil aux données")

if donnees:
    st.dataframe(data = df)
else:
    st.markdown("")

    

st.markdown('<h6 style="color:#9dacf6;"><u>Gestion des valeurs manquantes</u></h6>', unsafe_allow_html=True)

nan_val = st.button("Voir la répartition des NaN selon les variables")


if nan_val: 
    st.table(df.isnull().sum())
else:
    st.markdown("")
    
    
st.markdown("* Suppression des variables: *Rainfall*, *WindGustSpeed*, *Cloud9am*, *Cloud3pm*, *Pressure9am*, *Pressure3pm*, *Evaporation*, *Sunshine*.")

st.markdown("* Variables quantitatives: moyenne par station/mois/années via un regroupement de données.")

# 1/ SUPPRESSION DES VARIABLES PRESENTANT TROP DE NaN
df = df.drop(['Rainfall', 'WindGustSpeed', 'Cloud9am', 'Cloud3pm', 'Pressure9am', 'Pressure3pm', 
              'Evaporation', 'Sunshine'], axis = 1)

# 2/ VARIABLES RESTANTES: SUPPRESSION OU REMPLACEMENT DES NaN

# Suppression des NaN pour les variables 'RainToday' et 'RainTomorrow': 
df.dropna(axis = 0, how = 'any', subset = ['RainToday', 'RainTomorrow'], inplace = True)


# Pour le remplacement, on a besoin des dates (on supprimera ces colonnes par la suite): 
# Nouvelles variables pour les dates:
df['year'] = pd.to_datetime(df.Date).dt.year
df['month'] = pd.to_datetime(df.Date).dt.month
df['day'] = pd.to_datetime(df.Date).dt.day

df.drop('Date', axis = 1, inplace = True ) # suppression de la variable "Date" qui ne servira plus. 

# Min/MaxTemp; Temp9am/3pm; Humidity 9am/3pm: remplacement des NaN par moyenne par Station, par Mois, par Année:

mean_per_location = df.groupby(['Location','month', 'year']).mean().reset_index()

mean_loc = st.button("Voir le regroupement des données")
if mean_loc: 
    st.dataframe(data = mean_per_location)
else:
    st.markdown("")


for row in df.itertuples():
    # Si l'attribut MinTemp de la ligne row est nul: on va remplacer dans df, à l'index row, la valeur de MinTemp, par la moyenne calculée dans mean_per_location, pour la station/mois/année correspondante.
    if pd.isna(row.MinTemp):
        df.loc[row.Index, 'MinTemp'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['MinTemp'].values[0]
    
    # On applique le même raisonnement pour MaxTemp et autres...
    if pd.isna(row.MaxTemp):
        df.loc[row.Index, 'MaxTemp'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['MaxTemp'].values[0]
    
    if pd.isna(row.Temp9am):
        df.loc[row.Index, 'Temp9am'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['Temp9am'].values[0]
    
    if pd.isna(row.Temp9am):
        df.loc[row.Index, 'Temp3pm'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['Temp3pm'].values[0]
    
    if pd.isna(row.Humidity9am):
        df.loc[row.Index, 'Humidity9am'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['Humidity9am'].values[0]
    
    if pd.isna(row.Humidity3pm):
        df.loc[row.Index, 'Humidity3pm'] = mean_per_location[(mean_per_location['Location'] == row.Location) & (mean_per_location['month'] == row.month) & (mean_per_location['year'] == row.year)]['Humidity3pm'].values[0]
        
# Suppression des NaN restants pour toutes les variables:  
df.dropna(axis = 0, subset = ['Humidity9am', 'Humidity3pm', 'Temp3pm', 'WindSpeed9am', 'WindSpeed3pm','WindGustDir', 'WindDir9am', 'WindDir3pm'], inplace = True)



# BILAN CLEANING
# Stations supprimées: 
Loc = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree','Newcastle', 'NorahHead', 'NorfolkIsland', 
       'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 
       'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 
       'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 
       'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']

stat_erased = []

for loc in Loc:
    if loc not in df.Location.unique():
        stat_erased.append(loc)

df_saved = pd.read_csv("weatherAUS.csv")


# Suppression des variables relatives à la date ou à la station, car ne serviront pas pour le reste du notebook.
df.drop(['Location', 'year', 'month', 'day'], axis = 1, inplace = True)

col = st.button("Voir les variables restantes après cleaning")

if col:
    st.table(data = df.columns)
else:
    st.write("")

st.markdown('<h6 style="color:#9dacf6;"><u>Bilan cleaning</u></h6>', unsafe_allow_html=True)

st.write(f"* Nombre d'entrées finales, non nulles: {len(df)}")
st.write(f"* Deux stations supprimées: {stat_erased}")
st.write("* Conservation de 84% des données.")











## MACHINE LEARNING ##

st.markdown('<h2 style="color:#3adfb2;">2/ ML: RANDOM FOREST 🌧️🌳</h2>', unsafe_allow_html=True)

# PREPARATION POUR MACHINE LEARNING
st.markdown('<h6 style="color:#9dacf6;"><u>Préparation des données</u></h6>', unsafe_allow_html=True)

st.markdown("* Les données qualitatives sont encodées;")
st.markdown("* Les données sont séparées (variables explicatives VS target aka *RainTomorrow*);")
st.markdown("* Division en un ensemble d'apprentissage (80%) et un ensemble de test (20%);")

# Préparation des données
# RainToday et RainTomorrow:
df.replace(['Yes', 'No'], [1, 0], inplace = True)

# POUR LA DIRECTION DU VENT: 
wind = df['WindGustDir'].unique()

label = preprocessing.LabelEncoder() 

labels = label.fit_transform(wind)

df.replace(wind, labels, inplace = True)


# Séparation des données : 
data, target = df.drop('RainTomorrow', axis = 1), df['RainTomorrow']

# On sépare les données en un ensemble d'apprentissage et un ensemble de test, avec le ratio 80/20
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state=42)


st.markdown(" * Rééquilibrage (ou non) des données par Over et/ou Undersampling.")


# Rééquilibrage des données: 3 possibilités testées

over = RandomOverSampler(sampling_strategy = 0.6) # Fraction 60/40 
under = RandomUnderSampler() 

# a) Over puis under Sample
X_ov, y_ov = over.fit_resample(X_train, y_train) 

X_ov2, y_ov2 = over.fit_resample(X_train, y_train)
X_res, y_res = under.fit_resample(X_ov2, y_ov2) 


# b) Seulement un under Sample:
X_un, y_un = under.fit_resample(X_train, y_train)


donnees_choisies = st.selectbox(label = "Voir le résultat d'un rééquilibrage:", options=["Type de rééquilibrage", "Pas de rééquilibrage", "OverSampling", "UnderSampling", "Over + UnderSampling"])

if donnees_choisies == "Type de rééquilibrage":
    st.write("")
elif donnees_choisies == "Pas de rééquilibrage":
    st.write("Nombre d'entrées par classe:", y_train.value_counts(),"\nEn Proportion:", np.round(y_train.value_counts(normalize = True)*100, decimals = 1))
elif donnees_choisies == "OverSampling":
    st.write("Nombre d'entrées par classe:", y_ov.value_counts(),"\nProportion:", np.round(y_ov.value_counts(normalize = True)*100, decimals = 1))
elif donnees_choisies == "UnderSampling":
    st.write("Nombre d'entrées par classe:", y_un.value_counts(),"\nProportion:", np.round(y_un.value_counts(normalize = True)*100, decimals = 1))
else: st.write("Nombre d'entrées par classe:", y_res.value_counts(),"\nProportion:", np.round(y_res.value_counts(normalize = True)*100, decimals = 1))



# MODELES

# RANDOM FOREST
st.markdown('<h6 style="color:#9dacf6;"><u>Modèle(s) et Performances</u></h6>', unsafe_allow_html=True)

from sklearn.metrics import accuracy_score, f1_score, recall_score

st.markdown("Instanciation, entraînement et analyse des performances du modèle *Random Forest* sur les différentes préparations des données (pour comparaison).")

rf_donnees = st.selectbox(label = "Performances du Random Forest sur:", options = ['Type de données','Données non équilibrées', 'Données après OverSampling', 'Données après UnderSampling', 'Données après Over puis Undersampling'])



### TENTATIVE DE CACHE - NE FONCTIONNE PAS ##

#def train_model(rf_donnees):
#    if rf_donnees == "Type de données":    
#        return
#    elif rf_donnees == "Données non équilibrées":
#        data = non_balanced_data

#    st.write("Accuracy:", data["accuracy"])
#    st.write("f1_score:", data["f1_score"])
#    st.write("Rappel:", data["rappel"])
 
        


# @st.cache(suppress_st_warning=True)
# def train_model(rf_donnees):
#     print("coucou")
#     rf = RandomForestClassifier()
    
#     if rf_donnees == "Type de données":    
#         return
        
#     elif rf_donnees == "Données non équilibrées":
#         rf.fit(X_train, y_train)
        
#     elif rf_donnees == "Données après OverSampling":
#         rf.fit(X_ov, y_ov)

#     elif rf_donnees == "Données après UnderSampling":
#         rf.fit(X_un, y_un)

#     elif rf_donnees == "Données après Over puis Undersampling":
#         rf.fit(X_res, y_res)
    
#     y_pred = rf.predict(X_test)
    
#     #return 
#     st.write("Accuracy:", (accuracy_score(y_test, y_pred)*100).round())
#     st.write("f1_score:", (f1_score(y_test, y_pred)*100).round()) 
#     st.write("Rappel:", (recall_score(y_test, y_pred)*100).round())
    
#     matrice = st.button("Matrice de confusion")
#     if matrice:
#         st.table(pd.crosstab(y_test, y_pred, rownames = ['Classe réelle'], colnames = ['Classe prédite']))
    


# @st.cache
# def non_balanced_data():
#     rf = RandomForestClassifier()
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
    
#     return { "accuracy": (accuracy_score(y_test, y_pred)*100).round(), "f1_score": (f1_score(y_test, y_pred)*100).round(), "rappel": (recall_score(y_test, y_pred)*100).round() }
    

    

# if rf_donnees == "Type de données":
#     st.write("")
    
# elif rf_donnees == "Données non équilibrées":
#     rf1 = RandomForestClassifier()
#     rf1.fit(X_train, y_train)
#     y_pred1 = rf1.predict(X_test)
#     st.write("Accuracy:", (accuracy_score(y_test, y_pred1)*100).round())
#     st.write("F1_score:", (f1_score(y_test, y_pred1)*100).round()) 
#     st.write("Rappel:", (recall_score(y_test, y_pred1)*100).round())
    
# elif rf_donnees == "Données après OverSampling":
#     rf2 = RandomForestClassifier()
#     rf2.fit(X_ov, y_ov)
#     y_pred2 = rf2.predict(X_test)
#     st.write("Accuracy:", (accuracy_score(y_test, y_pred2)*100).round())
#     st.write("F1_score:", (f1_score(y_test, y_pred2)*100).round()) 
#     st.write("Rappel:", (recall_score(y_test, y_pred2)*100).round())
    
# elif rf_donnees == "Données après UnderSampling":
#     rf3 = RandomForestClassifier()
#     rf3.fit(X_un, y_un)
#     y_pred3 = rf3.predict(X_test)
#     st.write("Accuracy:", (accuracy_score(y_test, y_pred3)*100).round())
#     st.write("F1_score:", (f1_score(y_test, y_pred3)*100).round()) 
#     st.write("Rappel:", (recall_score(y_test, y_pred3)*100).round())
    
# elif rf_donnees == "Données après Over puis Undersampling":
#     rf4 = RandomForestClassifier()
#     rf4.fit(X_res, y_res)
#     y_pred4 = rf4.predict(X_test)
#     st.write("Accuracy:", (accuracy_score(y_test, y_pred4)*100).round())
#     st.write("F1_score:", (f1_score(y_test, y_pred4)*100).round()) 
#     st.write("Rappel:", (recall_score(y_test, y_pred4)*100).round())

if rf_donnees == "Type de données":
    st.write("")
    
elif rf_donnees == "Données non équilibrées":
    st.markdown("* Accuracy: 84%")
    st.markdown("* F1_score: 56%") 
    st.markdown("* Rappel: 45%")
    
elif rf_donnees == "Données après OverSampling":
    st.markdown("* Accuracy: 84%")
    st.markdown("* F1_score: 58%") 
    st.markdown("* Rappel: 51%")
    
elif rf_donnees == "Données après UnderSampling":
    st.markdown("* Accuracy: 78%")
    st.markdown("* F1_score: 60%") 
    st.markdown("* Rappel: 75%")
    
elif rf_donnees == "Données après Over puis Undersampling":
    st.markdown("* Accuracy: 82%")
    st.markdown("* F1_score: 61%") 
    st.markdown("* Rappel: 63%")



 
st.markdown('<h6 style="color:#9dacf6;"><u>Conclusion</u></h6>', unsafe_allow_html=True)     

st.markdown("On conserve le modèle le plus concluant: Random Forest sur données équilibrées par *over* puis *undersampling*.")


# MATRICE DE CONFUSION
rf_final = RandomForestClassifier()
rf_final.fit(X_res, y_res)
y_pred_final = rf_final.predict(X_test)

matrice = st.button("Matrice de confusion")
if matrice:
    st.table(pd.crosstab(y_test, y_pred_final, rownames = ['Classe réelle'], colnames = ['Classe prédite']))

st.markdown('<h6 style="color:#4500ef;"><i>⚠️ Les analyses suivantes sont donc effectuées après application du RF sur données rééquilibrées (over puis undersampling) :).</i></h6>', unsafe_allow_html=True)












## ANALYSES ##

st.markdown('<h2 style="color:#3adfb2;">3/ ANALYSES: FEATURE IMPORTANCE & COMPARAISON VP/FN</h2>', unsafe_allow_html=True)


st.markdown('<h6 style="color:#9dacf6;"><u>Feature Importance</u></h6>', unsafe_allow_html=True)

rf5 = RandomForestClassifier()
rf5.fit(X_res, y_res)
y_pred5 = rf5.predict(X_test)
    
importances = rf5.feature_importances_

impor=pd.DataFrame(data=(importances), index=data.columns, columns=['Importance'])
impor=impor.sort_values(by='Importance', ascending=False).T


st.table(impor.head(8))


st.markdown('<h6 style="color:#9dacf6;"><u>Comparaison VP versus FN</u></h6>', unsafe_allow_html=True)

st.markdown("Dans cette analyse, les prédictions sont comparées par rapport aux jours réels de pluie (RainTomorrow =1). Nous comparons les Vrais Positifs (bonne prédiction de pluie à j+1) aux Faux Négatifs (prédiction d'un jour sec à j+1 alors qu'il aura plu). Nous voulons voir, ne serait-ce que graphiquement, si certains patterns se dégagent de cette comparaison et pourraient nous aiguiller sur une amélioration des performances du modèle.")

# ANALYSE DES FN et FP

# y_pred to DataFrame, index of y_test to join to X_test df
predictions = pd.DataFrame(y_pred5).set_index(y_test.index)
predictions.columns = ['predictions']

df_rain = [y_test, predictions]

# New df with X_test, y_test & y_pred
df_compare = X_test.join(other = df_rain).reset_index()


df_compare.drop('index', axis = 1,  inplace = True)

# SUBDATAFRAMES:
VN = df_compare[(df_compare['RainTomorrow'] == 0) & (df_compare['predictions'] == 0)]
VP = df_compare[(df_compare['RainTomorrow'] == 1) & (df_compare['predictions'] == 1)] # intéressant à analyser
FP = df_compare[(df_compare['RainTomorrow'] == 0) & (df_compare['predictions'] == 1)] #intéressant à analyser
FN = df_compare[(df_compare['RainTomorrow'] == 1) & (df_compare['predictions'] == 0)]

FN_VP = pd.concat([FN, VP], axis = 0)


st.markdown('<h6 style="color:#4500ef;"><i>Comparaison des VP versus FN sur les variables quantitatives (températures, humidité, vitesse du vent):</i></h6>', unsafe_allow_html=True)

quantit = st.selectbox(label = "Choisissez la ou les variables à afficher:", options = ['Variable(s)','Températures', 'Humidité', 'Vitesse du vent'])

my_colors = ["#3adfb2", "#9dacf6"]
  
# add color array to set_palette
# function of seaborn
sns.set_palette( my_colors )

if quantit == 'Variable(s)':
    st.write("")
    
elif quantit == 'Températures':
    # TEMPERATURES
    fig1, axes = plt.subplots(2, 2, figsize=(35, 35))

    plt.subplot(221)
    sns.boxplot(ax=axes[0, 0], data=FN_VP, x='predictions', y='MinTemp')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('MinTemp', fontsize=30)
    plt.yticks(size = 20);

    plt.subplot(222)    
    sns.boxplot(ax=axes[0, 1], data=FN_VP, x='predictions', y='MaxTemp')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('MaxTemp', fontsize=30)
    plt.yticks(size = 20);

    plt.subplot(223)
    sns.boxplot(ax=axes[1, 0], data=FN_VP, x='predictions', y='Temp9am')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('Temp9am', fontsize=30)
    plt.yticks(size = 20);

    plt.subplot(224)
    sns.boxplot(ax=axes[1, 1], data=FN_VP, x='predictions', y='Temp3pm');
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('Temp3pm', fontsize=30)
    plt.yticks(size = 20);

    st.pyplot(fig1)
    st.markdown("Les VP montrent des valeurs de températures globalement plus basses que les FN. \nLes températures maximales semblent biaiser le modèle vers la prédiction d'un temps sec.")

elif quantit == "Humidité":
    #HUMIDITÉ
    fig2, axes = plt.subplots(1, 2, figsize=(35, 23))

    plt.subplot(121)
    sns.boxplot(data=FN_VP, x='predictions', y='Humidity9am')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('Humidity9am', fontsize=30)
    plt.yticks(size = 20);

    plt.subplot(122)
    sns.boxplot(data=FN_VP, x='predictions', y='Humidity3pm')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('Humidity3pm', fontsize=30)
    plt.yticks(size = 20);
    
    st.pyplot(fig2)
    st.markdown("Les FN montrent des taux d'humidité plus bas que les VP. La faible humidité semble donc biaiser le modèle vers la prédiction d'un temps sec.\nDe plus il y a un grand nombre de valeurs extrêmes/aberrantes.")

else:
    #VITESSE DU VENT
    fig3, axes = plt.subplots(1, 2, figsize=(35, 23))
    plt.subplot(121)
    sns.boxplot(data=FN_VP, x='predictions', y='WindSpeed9am')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('WindSpeed9am', fontsize=30)
    plt.yticks(size = 20);

    plt.subplot(122)
    sns.boxplot(data=FN_VP, x='predictions', y='WindSpeed3pm')
    plt.xticks(size = 20)
    plt.xlabel('Prédictions \n(valeur réelle = 1, soit "pluie le lendemain")', fontsize=30)
    plt.ylabel('WindSpeed3pm', fontsize=30)
    plt.yticks(size = 20);
    
    st.pyplot(fig3)
    st.markdown("Les VP semblent montrer des vents plus rapides que les FN.\nLes faibles vitesses de vents pourraient donc biaiser le modèle vers la prédiction d'un temps sec.\nDe plus nous remarquons un certain nombre de valeurs extrêmes/aberrantes.")
    

ccl1 = st.button("Conclusion variables quantitatives")
if ccl1:
    st.markdown("Globalement nous pouvons remarquer que le comportement de certaines variables peut biaiser les résultats du modèle, ici en faveur de la prédiction d'un temps sec.\nNous remarquons également un certain nombre de valeurs extrêmes pour chacune de ces variables qui pourraient empêcher le modèle de dégager des tendances pour les prédictions.\nNous voyons donc ici les limites du set de données malgré le cleaning.")



# CONCERNANT LES VENTS: 
st.markdown('<h6 style="color:#4500ef;"><i>Comparaison des VP versus FN sur les variables qualitatives (direction des bourrasques de vent, direction du vent à 9h et 15h):</i></h6>', unsafe_allow_html=True)
    
VP1 = FN_VP[FN_VP['predictions'] == 1]
FN1 = FN_VP[FN_VP['predictions'] == 0]


# WINDGUSTDIR
bins = range(0, 17)
ticks = range(0, 16)
    
vents = st.selectbox(label = "Choisissez la variable à afficher:", options = ['Variable','Direction des bourrasques', 'Direction du vent à 9h', 'Direction du vent à 15h'])


if vents == "Variable":
    st.write("")
    
elif vents == "Direction des bourrasques":
    fig4, axes = plt.subplots(1, 2, figsize=(40, 17))
   
    plt.subplot(121)
    plt.hist(VP1['WindGustDir'], bins = bins, color = 'b', edgecolor='grey', label = "VP WindGustDir"); #rwidth = 0.8
    plt.xticks(ticks, wind, rotation = 50);
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    plt.subplot(122)
    plt.hist(FN1['WindGustDir'], bins = range(0, 17), color = 'c', edgecolor='grey', label = 'FN WindGustDir'); #rwidth = 0.8
    plt.xticks(range(0, 16), wind, rotation = 50);
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    st.pyplot(fig4)
    st.markdown("Les bourrasques de vents Est, Est-Sud-Est et Sud-Sud-West notamment, semblent plus fréquentes et plus fortes pour les VP par rapport aux FN.")

elif vents == "Direction du vent à 9h":
    # WindDir9am
    fig5, axes = plt.subplots(1, 2, figsize=(40, 17))
    
    plt.subplot(121)
    plt.hist(VP1['WindDir9am'], bins = range(0, 17), color = 'b', edgecolor='grey', label = 'VP WindDir9am'); #rwidth = 0.8
    plt.xticks(range(0, 16), wind, rotation = 50)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    plt.subplot(122)
    plt.hist(FN1['WindDir9am'], bins = range(0, 17), color = 'c', edgecolor='grey', label = 'FN WindDir9am'); #rwidth = 0.8
    plt.xticks(range(0, 16), wind, rotation = 50)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    st.pyplot(fig5)
    st.markdown("Le Vent Nord-Est matinal est beaucoup plus fréquent et ressort d'autant plus pour les VP par rapport aux FN.")


else:
    fig6, axes = plt.subplots(1, 2, figsize=(40, 17))
    
    plt.subplot(121)
    plt.hist(VP1['WindDir3pm'], bins = range(0, 17), color = 'b', edgecolor='grey', label = 'VP WindDir3pm'); #rwidth = 0.8
    plt.xticks(range(0, 16), wind, rotation = 50)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    plt.subplot(122)   
    plt.hist(FN1['WindDir3pm'], bins = range(0, 17), color = 'c', edgecolor='grey', label = 'FN WindDir3pm'); #rwidth = 0.8
    plt.xticks(range(0, 16), wind, rotation = 50)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend(prop={'size': 20});

    st.pyplot(fig6)
    st.markdown("Plusieurs vents d'Est en après-midi semblent plus fréquents pour les Vrais Positifs.")


ccl2 = st.button("Conclusion variables qualitatives (vents)")
if ccl2:
    st.markdown("Il semble que certaines directions soient plus fréquentes pour les VP par rapport aux FN. Une moindre fréquence de ces vents semble donc biaiser le modèle en faveur de la prédiction d'un temps sec.")
else:
    st.write("")




ccl3 = st.button("CONCLUSION ANALYSES")
if ccl3:
    st.markdown('<h6 style="color:#9dacf6;"><u>CONCLUSION ANALYSES</u></h6>', unsafe_allow_html=True)
    st.markdown("Pour plus de rigueur, des tests statistiques devraient être réalisés pour mesurer la significativité des différences entre VP et FN.\nUne analyse des VN et FP aurait aussi pu être réalisée.\nSuite à cette analyse, de nouvelles variables ont été créées (différences de températures, d'humidité...) mais n'ont pas amélioré, voire ont diminué la performance du modèle.")
    st.markdown("Par la suite, nous avons tenté de récupérer des variables jusqu'alors supprimées, en améliorant l'étape de cleaning via le regroupement des stations par exemple.")
else:
    st.write("")













