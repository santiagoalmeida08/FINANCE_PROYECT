#Paquetes
import pandas as pd


#Importar datos 
historic = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv')
creditos = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv')

historic.columns
creditos.columns