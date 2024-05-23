import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score

#Funcion de encoding
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
  
  
#Evaluacion de modelos TRAIN-TEST
def modelos(list_mod, xtrain, ytrain, xtest, ytest):
    metrics_mod = pd.DataFrame()
    list_train = []
    list_test = []
    for modelo in list_mod:
        modelo.fit(xtrain,ytrain)
        y_pred = modelo.predict(xtest)
        score_train = metrics.mean_squared_error(ytrain,modelo.predict(xtrain)) #metrica entrenamiento  
        score_test = metrics.mean_squared_error(ytest,y_pred) #metrica test
        z= ['mod_lin','mod_rf','mod_gb']
        modelos = pd.DataFrame(z)
        list_test.append(np.sqrt(score_test)) #RSME
        list_train.append(np.sqrt(score_train)) #RSME
        pdscores_train = pd.DataFrame(list_train)
        pdscroestest = pd.DataFrame(list_test)
        
        metrics_mod = pd.concat([modelos, pdscores_train, pdscroestest], axis=1)
        metrics_mod.columns = ['modelo','score_train','score_test']
    return metrics_mod


#Evaluacion de modelos CROSS-VALIDATION

def medir_modelos(modelos,scoring,X,y,cv):
    "Recibe como parametros una lista de modelos, la metrica con la cual se quiere evaluar su desempeño, la base de datos escalada y codificada, la variable objetivo y el numero de folds para la validación cruzada."
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["rl","rf","gb"]#,"bagging"
    return metric_modelos   



#Funcion de transformacion de datos nuevos 
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
  return df

