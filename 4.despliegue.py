
import joblib 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

mod_final = joblib.load("salidas\\mod_final.pkl")
var_sel = joblib.load("salidas\\var_sel.pkl")
scaler = joblib.load("salidas\\scaler.pkl")
list_label = joblib.load("salidas\\list_label.pkl")
list_dumies = joblib.load("salidas\\list_dumies.pkl")

#Cargar datos

df = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv')
df.columns
#Transformación de datos para realizar predicciones

def transformar(df):
    df['NumberOfDependents'] = df['NumberOfDependents'].replace({0:'bajo', 1:'bajo',2:'bajo',
                                                                        3:'medio',4:'medio',5:'medio',6:'medio',
                                                                        7:'alto',8:'alto',9:'alto'})

    #Agrupar el numero de creditos en categorias
    df['NumberOfOpenCreditLinesAndLoans'] = df['NumberOfOpenCreditLinesAndLoans'].replace({0:'menos de 5', 1:'menos de 5',2:'menos de 5',3:'menos de 5',4:'menos de 5',5:'menos de 5'
                                                                        ,6:'entre 5-15',7:'entre 5-15',8:'entre 5-15',9:'entre 5-15',10:'entre 5-15',11:'entre 5-15',12:'entre 5-15',13:'entre 5-15',14:'entre 5-15',15:'entre 5-15'
                                                                        ,16:'mas de 15',17:'mas de 15',18:'mas de 15',19:'mas de 15'
                                                                        })
    #Agrupar el numero de veces en mora 

    df['NumberOfTimesPastDue'] = df['NumberOfTimesPastDue'].replace({0:'menos de 5', 1:'menos de 5',2:'menos de 5',3:'menos de 5',4:'menos de 5',5:'menos de 5'
                                                                        ,6:'entre 5-15',7:'entre 5-15',8:'entre 5-15',9:'entre 5-15',10:'entre 5-15',11:'entre 5-15',12:'entre 5-15',13:'entre 5-15',14:'entre 5-15',15:'entre 5-15'
                                                                        ,16:'mas de 15',17:'mas de 15',18:'mas de 15',19:'mas de 15'
                                                                        })



    #Agrupar employmentLenght en categorias

    df['EmploymentLength'] = df['EmploymentLength'].replace({0 : 'menos de 10', 1 : 'menos de 10', 2 : 'menos de 10', 3 : 'menos de 10', 4 : 'menos de 10', 5 : 'menos de 10', 6 : 'menos de 10', 7 : 'menos de 10', 8 :'menos de 10',9:'menos de 10',
                                                                    10:'entre 10-20',11:'entre 10-20',12:'entre 10-20',13:'entre 10-20',14:'entre 10-20',15:'entre 10-20',16:'entre 10-20',17:'entre 10-20',18:'entre 10-20',19:'entre 10-20',20:'entre 10-20',
                                                                    21:'mas de 20',22:'mas de 20',23:'mas de 20',24:'mas de 20',25:'mas de 20',26:'mas de 20',27:'mas de 20',28:'mas de 20',29:'mas de 20',30:'mas de 20'})


    #Agrupar years in current adress en categorias

    df['YearsAtCurrentAddress'] = df['YearsAtCurrentAddress'].replace({0 : 'menos de 10', 1 : 'menos de 10', 2 : 'menos de 10', 3 : 'menos de 10', 4 : 'menos de 10', 5 : 'menos de 10', 6 : 'menos de 10', 7 : 'menos de 10', 8 :'menos de 10',9:'menos de 10',
                                                                    10:'entre 10-20',11:'entre 10-20',12:'entre 10-20',13:'entre 10-20',14:'entre 10-20',15:'entre 10-20',16:'entre 10-20',17:'entre 10-20',18:'entre 10-20',19:'entre 10-20',20:'entre 10-20',
                                                                    21:'mas de 20',22:'mas de 20',23:'mas de 20',24:'mas de 20',25:'mas de 20',26:'mas de 20',27:'mas de 20',28:'mas de 20',29:'mas de 20',30:'mas de 20'})


    #Crear la variable objetivo con el complemento 
    """Como se quiere saber cual es porcentaje de pago de los usuarios y se tiene el de no pago esta nueva variable 
    que sera nuestra variable respuesta se haya restandole a 1 el porcentaje de no pago"""
    #df['Percent_paid'] = 1-df['NoPaidPerc']

    #Se puede sacar el valor de la deuda 
    df['Deuda'] = df['DebtRatio']*df['Assets'] 
    df['Deuda'] = df['Deuda'].round(2)
    
    df = df.drop(columns=['ID']) 
    df = df.drop(columns = ['NewLoanApplication'])
    return df



df_trans = transformar(df)


#Encoding
def encode_data(df, list_le, list_dd): 
    df_encoded = df.copy()   
    "Recibe como parametros la base de datos y las listas de variables que se quieren codificar"
    #Get dummies
    df_encoded=pd.get_dummies(df_encoded,columns=list_dd)
    
    # Ordinal Encoding
    #oe = OrdinalEncoder()
    #for col in list_oe:
    #    df_encoded[col] = oe.fit_transform(df_encoded[[col]])
    
    # Label Encoding
    le = LabelEncoder()
    for col in list_le:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

df_encoded = encode_data(df_trans,list_label, list_dumies)

# Escalado 
df_esc = scaler.transform(df_encoded)
df_esc = pd.DataFrame(df_esc, columns=df_encoded.columns)
df_escc = df_esc[var_sel]


#Prediccion de porcentaje de pago 
prediccion= mod_final.predict(df_escc)
pd_pred=pd.DataFrame(prediccion, columns=['Percent_paid'])
pred=pd.concat([df['ID'],pd_pred],axis=1)

pred['Percent_paid'].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

"""Partiendo de la distribución del porcentaje de pago los grupos para asignación de interes con el respectivo interes son:
-del 70-80% o menor al 80% = 7%
-del 81-85%= 5%
-86-90%=4%
-91-95%=3%
-96-100%= 2%
"""

interes= pred.copy()
def calcularinteres(Percent_paid):
    interesoperativo= 0.05
    interesdemargen= 0.1
    if Percent_paid <= 0.8:
        return 0.07+interesoperativo+interesdemargen
    elif 0.8 < Percent_paid <= 0.85:
        return 0.05+interesoperativo+interesdemargen
    elif 0.85 < Percent_paid <= 0.9:
        return 0.04+interesoperativo+interesdemargen
    elif 0.9 < Percent_paid <=0.95:
        return 0.03+interesoperativo+interesdemargen
    else:
        return 0.02+interesoperativo+interesdemargen


interes['Interes']=interes['Percent_paid'].apply(calcularinteres)

Interes_final= interes.drop(columns=['Percent_paid'])
Interes_final.to_excel("salidas\\Interes_final.xlsx")  #Exportar todas las  predicciones 
