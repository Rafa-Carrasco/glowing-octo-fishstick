from utils import db_connect
engine = db_connect()

# your code here
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. descargar data

# url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
# respuesta = requests.get(url)
# nombre_archivo = "medical_insurance_cost.csv"
# with open(nombre_archivo, 'wb') as archivo:
#     archivo.write(respuesta.content)


# 2. convertir csv en dataframe

total_data = pd.read_csv("../data/raw/medical_insurance_cost.csv")
total_data.shape
total_data.info()
total_data.head()

# buscar duplicados

total_data_sin = total_data.drop_duplicates()
total_data_sin.shape
# print(total_data.columns)

# Limpieza de datos
# - No hay duplicados ni valores nulos
# - Ninguna variables es objetivamente eliminable de entrada
# - 4 variables numericas y 3 categoricas

fig, axis = plt.subplots(2, 3, figsize = (10, 7))

# Crear un histograma múltiple
sns.histplot(ax = axis[0, 0], data = total_data, x = "sex").set(ylabel = None)
sns.histplot(ax = axis[0, 1], data = total_data, x = "smoker").set(ylabel = None)
sns.histplot(ax = axis[0, 2], data = total_data, x = "region").set(ylabel = None)

plt.tight_layout()
plt.show()

# Analisis univariable categorico
# - numero de hombres y mujeres es muy similar 
# - 3/4 partes son no fumadores
# - la distribucion regional es muy similar, con una ligera mayoria de usuarios del South East  

fig, axis = plt.subplots(2, 4, figsize = (10, 7), gridspec_kw={'height_ratios': [6, 1]})

# Crear una figura múltiple con histogramas y diagramas de caja
sns.histplot(ax = axis[0, 0], data = total_data, x = "age").set(xlabel = None)
sns.boxplot(ax = axis[1, 0], data = total_data, x = "age")
sns.histplot(ax = axis[0, 1], data = total_data, x = "bmi").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = total_data, x = "bmi")
sns.histplot(ax = axis[0, 2], data = total_data, x = "children").set(xlabel = None)
sns.boxplot(ax = axis[1, 2], data = total_data, x = "children")
sns.histplot(ax = axis[0, 3], data = total_data, x = "charges").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 3], data = total_data, x = "charges")
# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

# De esta visualizacion podemos ver que:
# - age tiene una distribucion simetrica practicamente perfecta
# - bmi tiene un ligero sesgo hacia los valores mas bajos y algunos valores atipicos en los maximos
# - children : asimetrica negativa. valores menores estan muy concentrados
# - charges : asimetrica negativa. valores menores muy concentrados. Importante numero de valores atipicos.  

# analis multivariable numerico entre predictoras y target (1)

fig, axis = plt.subplots(2, 3, figsize = (10, 7))

# Crear un diagrama de dispersión múltiple
sns.regplot(ax = axis[0, 0], data = total_data, x = "age", y = "charges")
sns.heatmap(total_data[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = total_data, x = "bmi", y = "charges").set(ylabel=None)
sns.heatmap(total_data[["charges", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])
sns.regplot(ax = axis[0, 2], data = total_data, x = "children", y = "charges").set(ylabel=None)
sns.heatmap(total_data[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 2])

# Ajustar el layout
plt.tight_layout()

# Mostrar el plot
plt.show()

# Analisis multivariable numerico-numerico (1)
# - correlacion directa debil fuerte entre age y target 
# - correlacion directa mas debil entre bmi y el target
# - correlacion muy debil entre numero de hijos y target

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[['age', 'bmi', 'children', 'charges']].corr(), annot = True, fmt = ".2f")

plt.tight_layout()
plt.show()

# las variables bmi y age son las que tienen una correlacion directa mas alta con el objetivo.
# Las demas tienen una correlacion muy debil con el target e incluso entre ellas. La unica reseñable sería bmi con age


total_data["sex_n"] = pd.factorize(total_data["sex"])[0]
total_data["smoker_n"] = pd.factorize(total_data["smoker"])[0]
total_data["region_n"] = pd.factorize(total_data["region"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[['age', 'sex_n', 'bmi', 'children', 'smoker_n', 'region_n', 'charges']].corr(), annot = True, fmt = ".2f")

plt.tight_layout()
plt.show()

# Análisis numérico-categórico (completo) 
# - no revela ninguna otra correlacion directa reseñable que las ya conocidas. 
# - sorprende la fuerte correlacion negativa de smoker con el objetivo.

# analisis de outliers

total_data.describe()
fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = total_data, y = "age")
sns.boxplot(ax = axis[0, 1], data = total_data, y = "sex_n")
sns.boxplot(ax = axis[0, 2], data = total_data, y = "bmi")
sns.boxplot(ax = axis[1, 0], data = total_data, y = "children")
sns.boxplot(ax = axis[1, 1], data = total_data, y = "smoker_n")
sns.boxplot(ax = axis[1, 2], data = total_data, y = "region_n")
sns.boxplot(ax = axis[2, 0], data = total_data, y = "charges")

plt.tight_layout()
plt.show()

# la variable charges (objetivo) es la que mas outliers tiene. 
# Mientras que el 75% de los valores se acercan a la media (13270), el maximo llega a los 63770. 
# Reemplazamos valores usando upper y lower limit.

charges_iqr = charges_stats["75%"] - charges_stats["25%"]
upper_limit = charges_stats["75%"] + 1.5 * charges_iqr
lower_limit = charges_stats["25%"] - 1.5 * charges_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}")

total_data[total_data["charges"] > 35000]

# filtramos los valores superiores al upper limit

total_data_fil = total_data[total_data["charges"] <= 35000]

# eliminar outliers para bmi

bmi_iqr = bmi_stats["75%"] - bmi_stats["25%"]
upper_limit = bmi_stats["75%"] + 1.5 * bmi_iqr
lower_limit = bmi_stats["25%"] - 1.5 * bmi_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers en la variable BMI son {round(upper_limit, 2)} y {round(lower_limit, 2)}")

# pero segun la World Health Organization (WHO)  UN bmi ≥ 40 EQUIVALE A Morbid Obesity/Class III 
# print(total_data_fil.columns) 

# filtramos los valores de bmi superiores a 40 

total_data_fil2 = total_data_fil[total_data_fil["bmi"] < 40]

total_data_fil2 = total_data_fil.drop(columns=['smoker', 'region', 'sex', 'region_n', 'sex_n'])
total_data_fil2.shape

from sklearn.model_selection import train_test_split

# Dividimos el conjunto de datos en muestras de train y test
X = total_data_fil2.drop("charges", axis = 1)
y = total_data_fil2["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.head()

# escalamiento min-max
 
from sklearn.preprocessing import MinMaxScaler

col_names = ['age', 'bmi', 'children', 'smoker_n']
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns=col_names)
X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns=col_names)
X_train_scal.head()

# guardar datos en csv

X_train_scal.to_csv("../data/interim/medical_insu_train.csv", index=False)
X_test_scal.to_csv("../data/interim/medical_insu_test.csv", index=False)

# entrenar modelo 

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scal, y_train)

print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b1, b2): {model.coef_}")

# prediccion 

y_pred = model.predict(X_test_scal)
y_pred

# metricas

from sklearn.metrics import mean_squared_error, r2_score

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")

# # optimizacion con lasso 

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

lasso_model = Lasso(alpha=0.2)  # alpha es el parámetro de regularización

# Entrenar el modelo
lasso_model.fit(X_train_scal, y_train)

# Hacer predicciones
y_pred = lasso_model.predict(X_test_scal)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuasdratico medio (Lasso):", mse)
print("Coeficientes (Lasso):", lasso_model.coef_)
print("Intercepto (Lasso):", lasso_model.intercept_)

# optimizacion con ridge 

from sklearn.linear_model import Ridge

# Entrenar el modelo de regresión Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scal, y_train)
y_pred = ridge_model.predict(X_test_scal)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)
