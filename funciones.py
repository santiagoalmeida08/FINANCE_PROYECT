import pandas as pd

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

