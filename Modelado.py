
# Realizar seleccion de variables y modelado 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#preprocesamiento
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
#Seleccion de variables
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso
#modelado
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


#Cargar datos

data = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Analitica3\\FINANCE_PROYECT\\base_full.csv')

#Pruba modelado usando train-test split y todas las variables 
#division
data = data.drop(columns=['NoPaidPerc']) # la eliminamos ya que esta variable podria afectar el rendimiento

datacross = data.copy()

x = data.drop(columns=['Percent_paid'])
y = data['Percent_paid']

list_dumies = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) > 2]
#list_ordinal = ['NumberOfOpenCreditLinesAndLoans','NumberOfTimesPastDue','EmploymentLength','YearsAtCurrentAddress'	]
list_label = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) == 2]


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

df_encoded = encode_data(x, list_label, list_dumies)

#Escalado 
scaler = StandardScaler()
x_esc = scaler.fit_transform(df_encoded)

#Division de datos train/test

xtrain,xtest,ytrain,ytest = train_test_split(x_esc,y,test_size=0.2,random_state=42)

#Evaluacion modelos 
mod_lin = LinearRegression()
mod_rf = RandomForestRegressor()
mod_dt = DecisionTreeRegressor()

list_mod = [mod_lin,mod_rf,mod_dt]

def modelos(list_mod, xtrain, ytrain, xtest, ytest):
    metrics_mod = pd.DataFrame()
    list_train = []
    list_test = []
    for modelo in list_mod:
        modelo.fit(xtrain,ytrain)
        y_pred = modelo.predict(xtest)
        score_train = metrics.mean_squared_log_error(ytrain,modelo.predict(xtrain)) #metrica entrenamiento  
        score_test = metrics.mean_squared_log_error(ytest,y_pred) #metrica test
        z= ['mod_lin','mod_rf','mod_dt']
        modelos = pd.DataFrame(z)
        list_test.append(score_test)
        list_train.append(score_train)
        pdscores_train = pd.DataFrame(list_train)
        pdscroestest = pd.DataFrame(list_test)
        
        metrics_mod = pd.concat([modelos, pdscores_train, pdscroestest], axis=1)
        metrics_mod.columns = ['modelo','score_train','score_test']
    return metrics_mod

modelos(list_mod, xtrain, ytrain, xtest, ytest)


"""Se observa que se tiene buen desempeño en general, por lo cual procedemos a haceel modelado con cross-validaton
y seleccion de variables"""
#Carga de base de datos

datac = datacross.copy()

#Eliminamos variables innecesarias 

datac = datac.drop(columns = ['ID'])

#las listas de definieron en el modelado de prueba 
datac_encoded = encode_data(datac, list_label, list_dumies) # ajuste respecto al orden (ordinal encoding) de dependientes,direccion?

# division de datos

x = datac_encoded.drop(columns=['Percent_paid'])
y = datac_encoded['Percent_paid']

#Escalado de variables 
scaler = StandardScaler()
x_esc = scaler.fit_transform(x)
x_esc = pd.DataFrame(x_esc, columns = x.columns)


#Seleccion de variables 

sel_ = SelectFromModel(Lasso(alpha = 0.001, max_iter=10000), max_features=8) 
sel_.fit(x_esc, y)
print(sel_.estimator_.coef_)
#Obtener variables seleccionadas
x_new = sel_.get_support()#descarta los coeficientes mas cercanos a 0

df_new = x_esc.iloc[:,x_new]
df_new.head()

df_sel = df_new.copy() #Variables seleccionadas

# Algoritmos a modelar 
mod_lin = LinearRegression()
mod_rf = RandomForestRegressor()
mod_gb = GradientBoostingRegressor()
list = [mod_lin,mod_rf,mod_gb]

#Evaluacion del desempeño de modelos 

def medir_modelos(modelos,scoring,X,y,cv):
    "Recibe como parametros una lista de modelos, la metrica con la cual se quiere evaluar su desempeño, la base de datos escalada y codificada, la variable objetivo y el numero de folds para la validación cruzada."
    #os = RandomOverSampler() # Usamos random oversampling para balancear la base de datos ya que la variable objetivo esta desbalanceada
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        #pipeline = make_pipeline(os, modelo)
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["Lineal_Regres","random_forest","grad_boost"]
    return metric_modelos   


var_total = medir_modelos(list,"neg_root_mean_squared_error",x_esc,y,10) #RMSE
var_sel = medir_modelos(list,"neg_root_mean_squared_error",df_sel,y,10) #RMSE

df_eval = pd.concat([var_total,var_sel],axis=1)
df_eval.columns = ['Lineal_Regres','random_forest','grad_boost','Lineal_Regres_sel','random_forest_sel','grad_boost_sel']

df_eval.plot(kind='box', figsize=(10,6))
plt.title('Desempeño de modelos con todas las variables y variables seleccionadas')
df_eval.mean()

""" POR AHORA SE OBSERVA UN RENDIMIENTO ADECUADO, EL R2 NO ES MUY BUENO CON LAS VARIABLES SELECCIONADAS Y EL RMS TAMBIEN BAJA SU DESEMPEÑO EN COMPARACION DE LAS VARIABLES TOTALES
HAY QUE CONSIDERAR : 
- PONER MENOS PESO EN EL METODO DE SELECCION DE VARIABLES PARA QUE NO DESCARTE TANTAS VARIABLES
- EXPLORAR OTRO METODO DE SELECCION DE VARIABLES Y EVALUAR EL RENDIMIENTO CON ESAS VARIABLES
- EXPLORAR ALGORITMOS COMO XTREME GRADIENT BOOSTING O METODOS DE ENSAMBLE COMO BAGGING O BOOSTING
- NO ESTARIA MAL HACER UNA RED NEURONAL PARA MIRAR EL RENDIMIENTO"""    