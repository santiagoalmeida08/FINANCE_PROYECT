
# Realizar seleccion de variables y modelado 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#modelado_aprendizaje supervisado 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#funciones
import funciones as fn
#Ajuste de hiperparametros
from sklearn.model_selection import RandomizedSearchCV
#exportar modelo
import joblib


#Cargar datos

data = joblib.load('data\\base_full.pkl')

#Pruba modelado usando train-test split y todas las variables 
#division
data = data.drop(columns=['NoPaidPerc']) # la eliminamos ya que esta variable podria afectar el rendimiento

datacross = data.copy()

x = data.drop(columns=['Percent_paid'])
y = data['Percent_paid']

list_dumies = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) > 2]
#list_ordinal = ['NumberOfOpenCreditLinesAndLoans','NumberOfTimesPastDue','EmploymentLength','YearsAtCurrentAddress'	]
list_label = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) == 2]


df_encoded = fn.encode_data(x, list_label, list_dumies)

#Escalado 
scaler = StandardScaler()
x_esc = scaler.fit_transform(df_encoded)

#Division de datos train/test
xtrain,xtest,ytrain,ytest = train_test_split(x_esc,y,test_size=0.2,random_state=42)

#Evaluacion modelos 
mod_lin = LinearRegression()
mod_rf = RandomForestRegressor()
mod_gb = GradientBoostingRegressor(n_estimators=700)

list_mod = [mod_lin,mod_rf,mod_gb]

fn.modelos(list_mod, xtrain, ytrain, xtest, ytest)

mod_gb.fit(xtrain,ytrain)
ypredgb = mod_gb.predict(xtest)

# Metricas
print("MAPE: %.2f" % metrics.mean_absolute_percentage_error(ytest, ypredgb))
print("RMSE: %.2f" % np.sqrt(metrics.mean_squared_error(ytest, ypredgb)))
print("MAE: %.2f" % metrics.mean_absolute_error(ytest, ypredgb))


"""Se observa que se tiene buen desempeño en general, por lo cual procedemos a haceel modelado con cross-validaton"""
#Carga de base de datos

datac = datacross.copy()

#Eliminamos variables innecesarias 

datac = datac.drop(columns = ['ID'])

#las listas de definieron en el modelado de prueba 
datac_encoded = fn.encode_data(datac, list_label, list_dumies) # ajuste respecto al orden (ordinal encoding) de dependientes,direccion?

# division de datos

x = datac_encoded.drop(columns=['Percent_paid'])
y = datac_encoded['Percent_paid']

#Escalado de variables 
scaler = StandardScaler()
x_esc = scaler.fit_transform(x)
x_esc = pd.DataFrame(x_esc, columns = x.columns)

df_sel = x_esc.copy() #Variables seleccionadas
var_sel = df_sel.columns

# Algoritmos a modelar 
mod_lin = LinearRegression()
mod_rf = RandomForestRegressor(random_state=42)
mod_gb = GradientBoostingRegressor(n_estimators= 700 , random_state=42)
list = [mod_lin,mod_rf,mod_gb]

#Evaluacion del desempeño de modelos 
var_total = fn.medir_modelos(list,"neg_root_mean_squared_error",df_sel,y,10) #RMSE
df_eval = pd.concat([var_total],axis=1)
df_eval.columns = ['rl','rf','gb']
df_eval.plot(kind='box', figsize=(10,6))
plt.title('Desempeño de modelos con todas las variables y variables seleccionadas')
df_eval.mean()

#Ajuste de hiperparametros 
params = {'n_estimators' : [100,500,700],
          'learning_rate' : [0.1],
          'max_depth' : [3,5], # 'max_features' : 'auto
          'min_samples_split' : [2,4,6],
          'min_samples_leaf' : [20,40,60]}

h1 = RandomizedSearchCV(mod_gb, param_distributions=params, n_iter=10, cv=10, scoring='neg_root_mean_squared_error', random_state=42,n_jobs=-1)
h1.fit(df_sel,y)

resultados = h1.cv_results_
h1.best_params_
mod_final = h1.best_estimator_

#Evaluar mtrain-test por cross-validation
eval = cross_validate(mod_final,df_sel,y,scoring='neg_root_mean_squared_error',cv=5,return_train_score=True, n_jobs=-1)
train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']

print("TRAIN RSME: %2f" % train_test_rf['train_score'].mean())
print("TEST RSME: %2f" % train_test_rf["test_score"].mean())

#Analizar el error de los modelos ?


#Exportar elementos 
joblib.dump(list_dumies, 'salidas\\list_dumies.pkl')
joblib.dump(list_label, 'salidas\\list_label.pkl') 
joblib.dump(scaler, 'salidas\\scaler.pkl')
joblib.dump(var_sel, 'salidas\\var_sel.pkl')
joblib.dump(mod_final, 'salidas\\mod_final.pkl')


