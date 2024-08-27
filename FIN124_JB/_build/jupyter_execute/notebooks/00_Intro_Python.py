#!/usr/bin/env python
# coding: utf-8

# # Introducción Python

# ## ¿Por qué Python?
# 
# <p style="text-align: justify;">
# Python es uno de los lenguajes de programación más demandados en la industria, con aplicaciones en sectores como tecnología, finanzas, investigación científica y educación. Asimismo es un lenguaje de propósito general, lo que significa que se puede utilizar en una amplia gama de aplicaciones, desde desarrollo web y automatización hasta análisis de datos, inteligencia artificial, y desarrollo de juegos. </p>
# 
# - [Perú](https://trends.google.es/trends/explore?date=all&geo=PE&q=%2Fm%2F0212jm,%2Fm%2F05z1_,%2Fm%2F053_x,%2Fm%2F05ymvd,%2Fm%2F0b7gmz&hl=es)
# - [Mundo](https://trends.google.es/trends/explore?date=all&q=%2Fm%2F0212jm,%2Fm%2F05z1_,%2Fm%2F053_x,%2Fm%2F05ymvd,%2Fm%2F0b7gmz&hl=es)

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from IPython.display import Image;


# In[2]:


# Importar datos Perú
peru         = pd.read_excel(r".\_data\google_trends_peru.xlsx")
peru["date"] = pd.date_range(start='2004-01',
                           end='2024-09',
                           freq = "M")
peru         = peru.set_index("date")
peru         = peru.drop('Mes', axis=1)

# Importar datos Mundo
world = pd.read_excel(r".\_data\google_trends_global.xlsx")
world["date"] = pd.date_range(start='2004-01',
                           end='2024-09',
                           freq = "M")
world = world.set_index("date")
world = world.drop('Mes', axis=1)

# Interfaz del gráfico
fig, axs = plt.subplots(2,1, figsize=(14, 8))

axs[0].plot(peru)
axs[0].grid(which='major')
axs[0].set_title("Interés por lenguajes de programación - Perú")
axs[0].legend(['R', 'Python', 'Matlab', 'Stata', 'Eviews'])

axs[1].plot(world)
axs[1].grid(which='major')
axs[1].set_title("Interés por lenguajes de programación - Mundo")
axs[1].legend(['R', 'Python', 'Matlab', 'Stata', 'Eviews'])


# ## ¿Cómo descargar Python?
# 
# 
# Para descargar e instalar Python a través de la distribución Anaconda, sigue estos pasos:
# 
# __1. Visita el sitio web de Anaconda__
# 
# - Abre tu navegador web y ve a la página oficial de Anaconda: [anaconda.com](https://www.anaconda.com/download).
# 
# __2. Descargar Anaconda__
# 
# - En la página de Anaconda, encontrarás un casillero para colocar tu correo, al cuál te llegará un link y podrás iniciar la descarga.
# 
# - Por defecto te enviará a la versión compatible con tu sistema operativo, de igual manera puedes elegir la versión que mejor te funcione
# 
# __3. Instalar Anaconda__
# 
# - Una vez que la descarga esté completa, abre el archivo del instalador.
# - Sigue las instrucciones del asistente de instalación.
# - Si durante el proceso de instalación te preguntan si deseas hacer de Anaconda tu instalación predeterminada de Python, selecciona "sí".
# 
# __4. Uso de Jupyther Notebooks__
# 
# - Una vez instalada anaconda se habrá instalado tambien una aplicación llamada `Jupyther Notebook`, la cual puede buscar en su menú de aplicaciones o a través del buscador de su barra de tareas
# 
# - Al abrirlo lanzará el programa (terminal)

# In[3]:


Image("./_images/terminal.jpeg")


# - Esto tambien abriará nuestro navegador con una interfaz más amigable, con la cual podremos elegir carpetas donde crear nuestro primer notebook haciendo click en "New" y en "Python 3"

# In[4]:


Image("./_images/jupyther.png")


# - Con ello estaremos listos para el primer `Hola Mundo`

# In[5]:


Image("./_images/notebook.png")


# ## Hola Mundo
# 
# En los Jupyther Notebooks podemos diferencias dos tipos primarios de celdas, una celda Código y una celda Markdown. En la primera podremos escribir comandos de python y en la segunda texto, al que le podemos dar formato HTML 

# In[6]:


print('Hola Mundo')


# Hola Mundo

# ## Tipos de datos en Python
# 
# En Python, existen varios tipos de datos incorporados que se pueden utilizar para almacenar y manipular información. A continuación se presentan los principales tipos de datos:
# 
# __1. Tipos de datos numéricos__
# 
# - Enteros (int): Representan números enteros, positivos o negativos, sin parte decimal.

# In[7]:


type(10)


# - Flotantes (float): Representan números con punto decimal.

# In[8]:


type(10.25)


# - Números complejos (complex): Representan números complejos con parte real e imaginaria.

# In[9]:


type(4+2j)


# __2. Cadenas de texto (str)__
# 
# Representan secuencias de caracteres que se pueden definir usando comillas simples (') o dobles (").

# In[10]:


type("Hola Mundo")


# In[11]:


type('python')


# __3. Booleanos (bool)__
# 
# Representan valores de verdad: True o False, se utilizan comúnmente en condiciones y comparaciones.

# In[12]:


1==1


# In[13]:


type(1==1)


# In[14]:


1==0


# In[15]:


type(1==0)


# __4. Listas (list)__
# 
# Son colecciones ordenadas de elementos que pueden ser de diferentes tipos de datos. Se definen usando corchetes. Estas son mutables

# In[16]:


type([1, 2, 3, 4])


# In[17]:


lista_1 = [1, 2, 3, 4]
lista_1[0] = "ola"
lista_1


# __5. Tuplas (tuple)__
# 
# Son colecciones ordenadas de elementos, similares a las listas, pero son inmutables (no se pueden modificar después de su creación). Se definen usando paréntesis.

# In[18]:


type((1, 2, 3, 4))


# In[19]:


tupla_1 = (1, 2, 3, 4)
tupla_1[0] = "ola"


# __6. Conjuntos (set)__
# 
# Son colecciones desordenadas de elementos únicos. Se definen usando llaves ({}), pero no permiten elementos duplicados.

# In[20]:


{1,1,2,3,4,4,4}


# In[21]:


type({1,1,2,3,4,4,4})


# __7. Diccionarios (dict)__
# 
# Son colecciones desordenadas de pares clave-valor.Se definen usando llaves, donde cada elemento tiene una clave y un valor.

# In[22]:


{"nombre": "Juan", "edad": 30}


# In[23]:


type({"nombre": "Juan", "edad": 30})


# __8. DataFrame__
# 
# <p style="text-align: justify;">
# Un DataFrame es una estructura de datos bidimensional en Python, que es parte de la biblioteca Pandas. Se puede pensar en un DataFrame como una tabla, similar a una hoja de cálculo de Excel o una tabla en una base de datos, donde los datos se organizan en filas y columnas.
# </p>
# 
# Características principales de un DataFrame:
# 
# - Estructura Tabular: Los datos están organizados en filas y columnas, donde cada columna puede contener diferentes tipos de datos (enteros, flotantes, cadenas de texto, etc.).
# 
# 
# - Etiquetas de Filas y Columnas: Las filas y columnas de un DataFrame tienen etiquetas, lo que permite acceder a los datos de manera eficiente. Las etiquetas de las filas se llaman índices, mientras que las etiquetas de las columnas se corresponden con los nombres de las columnas. 
# 
# 
# - Operaciones de Datos: Pandas ofrece una variedad de funciones y métodos para realizar operaciones de datos en DataFrames, como filtrado, agrupamiento, agregación, fusiones y transformaciones.
# 
# 
# - Datos Faltantes: Los DataFrames pueden manejar datos faltantes de manera eficiente, permitiendo identificar, rellenar o eliminar estos datos según sea necesario.

# In[24]:


import pandas as pd

# Crear un DataFrame a partir de un diccionario
datos = {
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María'],
    'Edad': [23, 45, 31, 27],
    'Ciudad': ['Lima', 'Arequipa', 'Cajamarca', 'Piura']
}

df = pd.DataFrame(datos)

# Mostrar el DataFrame
df


# In[25]:


type(df)


# __9. Date y DateTime__
# 
# `date` Contiene el formato de una fecha

# In[26]:


from datetime import date

hoy = date.today()
print(hoy)


# In[27]:


type(hoy)


# `datetime` Contiene mayor información adicional (timestamp) a la fecha

# In[28]:


from datetime import datetime

ahora = datetime.now()
print(ahora)


# In[29]:


type(ahora)


# <p style="text-align: justify;">
# Es importante diferenciar ambos tipos de datos pues al aplicar alguna operación o función (e.g. una proyección) a nuestros datos en formato `date` puede darnos por resultado un dato en formato `datetime`, lo que nos lleva a tener que transformar nuestros datos para hacerlos comparables, unirlos o aplicarles alguna otra operación.
# </p>
