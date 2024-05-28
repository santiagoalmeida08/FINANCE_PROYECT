
import joblib 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import funciones as fn

mod_final = joblib.load("salidas\\mod_final.pkl")
var_sel = joblib.load("salidas\\var_sel.pkl")
scaler = joblib.load("salidas\\scaler.pkl")
list_label = joblib.load("salidas\\list_label.pkl")
list_dumies = joblib.load("salidas\\list_dumies.pkl")

#Cargar datos

df = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv')
df.columns

#Transformación de datos para realizar predicciones
df_trans = fn.preparar_data(df)

#Encoding
df_encoded = fn.encode_data(df_trans,list_label, list_dumies)

# Escalado 
df_esc = scaler.transform(df_encoded)
df_esc = pd.DataFrame(df_esc, columns=df_encoded.columns)
df_escc = df_esc[var_sel]


#Prediccion de porcentaje de pago 
prediccion= mod_final.predict(df_escc)
pd_pred=pd.DataFrame(prediccion, columns=['Percent_paid'])
pred=pd.concat([df['ID'],pd_pred],axis=1)

pred['Percent_paid'].hist(figsize=(8, 8), bins=50)

pred[pred['Percent_paid']>1.0].count()

"""Partiendo de la distribución del porcentaje de pago los grupos para asignación de interes con el respectivo interes son:
-menor a 70% = 12%
-70-80 = 7%
-del 81-85% = 5%
-86-90%=4%
-91-95%=3%
-96-100%= 2%
"""

interes= pred.copy()

interes['int_rc'] = 1-pred['Percent_paid'] + 0.05 # +- media del error

"""
def calcularinteres(Percent_paid):
    interesoperativo= 0.05
    interesdemargen= 0.1
    if Percent_paid <= 0.7:
        return 0.12+interesoperativo+interesdemargen
    elif 0.7<Percent_paid<= 0.8:
        return 0.07+interesoperativo+interesdemargen
    elif 0.8 < Percent_paid <= 0.85:
        return 0.05+interesoperativo+interesdemargen
    elif 0.85 < Percent_paid <= 0.9:
        return 0.04+interesoperativo+interesdemargen
    elif 0.9 < Percent_paid <=0.95:
        return 0.03+interesoperativo+interesdemargen
    else:
        return 0.02+interesoperativo+interesdemargen 
interes['Interes']=interes['Percent_paid'].apply(calcularinteres)"""


Interes_final = interes.drop(columns = 'Percent_paid')
Interes_final['int_rc'].hist(figsize=(8, 8), bins=50)
Interes_final.describe()

#Interes_final.to_excel("salidas\\Interes_final.xlsx")  #Exportar todas las  predicciones 
