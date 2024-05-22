#Objetivo analisis : estructurar, tranformar los datos para craear nuevas variables 

#Paquetes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importar datos 
historic = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv')
creditos = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv')

#Analisis de historico
historic.dtypes
historic.info()

historic['ID'].duplicated().sum()

base_full = historic.copy()

#Analisis variables numericas 

df_numeric = historic.select_dtypes(include=['float64', 'int64'])

df_numeric.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

df_numeric['NumberOfTimesPastDue'].value_counts()

correlation = df_numeric.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""Estructuramos los datos para hacer mas facil el analisis bivariado y creaun nuevas variables"""

"""En las graficas de distribución de los datos se observa que hay algunas variables que para mejorar 
su interpretabilidad es mejor agrupar los valores por categorias tal como se muestra acontinuacion"""
#Agrupar el numero de dependientes en categorias
base_full['NumberOfDependents'].value_counts()

base_full['NumberOfDependents'] = base_full['NumberOfDependents'].replace({0:'bajo', 1:'bajo',2:'bajo',
                                                                           3:'medio',4:'medio',5:'medio',6:'medio',
                                                                         7:'alto',8:'alto',9:'alto'})
base_full['NumberOfDependents'].value_counts()
base_full.columns

#Agrupar el numero de creditos en categorias
base_full['NumberOfOpenCreditLinesAndLoans'].value_counts()
base_full['NumberOfOpenCreditLinesAndLoans'] = base_full['NumberOfOpenCreditLinesAndLoans'].replace({0:'menos de 5', 1:'menos de 5',2:'menos de 5',3:'menos de 5',4:'menos de 5',5:'menos de 5'
                                                                           ,6:'entre 5-15',7:'entre 5-15',8:'entre 5-15',9:'entre 5-15',10:'entre 5-15',11:'entre 5-15',12:'entre 5-15',13:'entre 5-15',14:'entre 5-15',15:'entre 5-15'
                                                                           ,16:'mas de 15',17:'mas de 15',18:'mas de 15',19:'mas de 15'
                                                                         })
base_full['NumberOfOpenCreditLinesAndLoans'].value_counts()

#Agrupar el numero de veces en mora 
base_full['NumberOfTimesPastDue'].value_counts()

base_full['NumberOfTimesPastDue'] = base_full['NumberOfTimesPastDue'].replace({0:'menos de 5', 1:'menos de 5',2:'menos de 5',3:'menos de 5',4:'menos de 5',5:'menos de 5'
                                                                           ,6:'entre 5-15',7:'entre 5-15',8:'entre 5-15',9:'entre 5-15',10:'entre 5-15',11:'entre 5-15',12:'entre 5-15',13:'entre 5-15',14:'entre 5-15',15:'entre 5-15'
                                                                           ,16:'mas de 15',17:'mas de 15',18:'mas de 15',19:'mas de 15'
                                                                         })
base_full['NumberOfTimesPastDue'].value_counts()


#Agrupar employmentLenght en categorias
base_full['EmploymentLength'].value_counts()
base_full['EmploymentLength'] = base_full['EmploymentLength'].replace({0 : 'menos de 10', 1 : 'menos de 10', 2 : 'menos de 10', 3 : 'menos de 10', 4 : 'menos de 10', 5 : 'menos de 10', 6 : 'menos de 10', 7 : 'menos de 10', 8 :'menos de 10',9:'menos de 10',
                                                                       10:'entre 10-20',11:'entre 10-20',12:'entre 10-20',13:'entre 10-20',14:'entre 10-20',15:'entre 10-20',16:'entre 10-20',17:'entre 10-20',18:'entre 10-20',19:'entre 10-20',20:'entre 10-20',
                                                                       21:'mas de 20',22:'mas de 20',23:'mas de 20',24:'mas de 20',25:'mas de 20',26:'mas de 20',27:'mas de 20',28:'mas de 20',29:'mas de 20',30:'mas de 20'})
base_full['EmploymentLength'].value_counts()

#Agrupar years in current adress en categorias
base_full['YearsAtCurrentAddress'].value_counts()
base_full['YearsAtCurrentAddress'] = base_full['YearsAtCurrentAddress'].replace({0 : 'menos de 10', 1 : 'menos de 10', 2 : 'menos de 10', 3 : 'menos de 10', 4 : 'menos de 10', 5 : 'menos de 10', 6 : 'menos de 10', 7 : 'menos de 10', 8 :'menos de 10',9:'menos de 10',
                                                                       10:'entre 10-20',11:'entre 10-20',12:'entre 10-20',13:'entre 10-20',14:'entre 10-20',15:'entre 10-20',16:'entre 10-20',17:'entre 10-20',18:'entre 10-20',19:'entre 10-20',20:'entre 10-20',
                                                                       21:'mas de 20',22:'mas de 20',23:'mas de 20',24:'mas de 20',25:'mas de 20',26:'mas de 20',27:'mas de 20',28:'mas de 20',29:'mas de 20',30:'mas de 20'})
base_full['YearsAtCurrentAddress'].value_counts()

#Crear la variable objetivo con el complemento 
"""Como se quiere saber cual es porcentaje de pago de los usuarios y se tiene el de no pago esta nueva variable 
que sera nuestra variable respuesta se haya restandole a 1 el porcentaje de no pago"""
base_full['Percent_paid'] = 1-base_full['NoPaidPerc']

#Se puede sacar el valor de la deuda 
base_full['Deuda'] = base_full['DebtRatio']*base_full['Assets'] 
base_full['Deuda'] = base_full['Deuda'].round(2)


#Analisis de variables categoricas
df_cat = base_full.copy()

#numero de dependientes vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['NumberOfDependents'], y=df_cat['Percent_paid'])
plt.title('NumberOfDepends vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

"""Se puede observar que no se observa una relación directa entre el numero de dependientes 
y el pago de la deuda, se observan algunos valores atipicos tanto de una mayor como de un menor 
porcentaje de pago, siendo mayor el numero de atipicos de este ultimo para la categoria de un 
alto numero de dependientes"""


#numero de creditos vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['NumberOfOpenCreditLinesAndLoans'], y=df_cat['Percent_paid'])
plt.title('NumberOfOpenCreditLinesAndLoans vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#numero de veces en mora vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['NumberOfTimesPastDue'], y=df_cat['Percent_paid'])
plt.title('NumberOfTimesPastDue vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#EmploymentLength vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['EmploymentLength'], y=df_cat['Percent_paid'])
plt.title('EmploymentLength vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#YearsAtCurrentAddress vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['YearsAtCurrentAddress'], y=df_cat['Percent_paid'])
plt.title('YearsAtCurrentAddress vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#HomeOwnership vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['HomeOwnership'], y=df_cat['Percent_paid'])
plt.title('HomeOwnership vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#Education vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['Education'], y=df_cat['Percent_paid'])
plt.title('Education vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()

#MaritalStatus vs porcentaje pagado
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cat, x=df_cat['MaritalStatus'], y=df_cat['Percent_paid'])
plt.title('MaritalStatus vs Percent_paid')
plt.ylabel('Percent_page')
plt.show()


base_full.to_csv('base_full.csv', index=False)
