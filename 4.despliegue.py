
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

"""El interes se tomara teniendo encuenta el porcentaje de no no pago y se le sumara a 
este un paroximado del error, debido a que el modelo segun el grafico de los residuales 
no esta sobreestimando ni subestimando, el error se va a sumar ya que es mejor tener una 
ganancia a que el modelo subestime y se reste el error y se este teniendo una perdida
"""

interes= pred.copy()

interes['int_rc'] = 1-pred['Percent_paid'] + 0.05 # + media del error

Interes_final = interes.drop(columns = 'Percent_paid')
Interes_final['int_rc'].hist(figsize=(8, 8), bins=50)
Interes_final= Interes_final.round(2)
Interes_final.describe()

Interes_final.to_excel("salidas\\Interes_final.xlsx") #Exportar todas las  predicciones 
