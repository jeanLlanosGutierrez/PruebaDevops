#%%
import os
import pandas as pd
import numpy as np
import sys
import warnings
import subprocess
import easygui
import re

path_python = sys.executable
path_python = path_python.replace('\\','/')
warnings.filterwarnings("ignore")

# Para obtener la ruta del directorio donde se encuentra el script
ruta_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(ruta_script)
ruta_script = ruta_script.replace('\\','/')

# IMPUTS
cantidad_scrap = int(input("Número de registros a buscar:"))
bloque_busqueda = int(input("Número de bloque a buscar:"))
numero_procesadores = int(input("Número de procesadores:"))
ruta_parquet = easygui.fileopenbox(msg="Seleccionar el archivo de base de búsqueda.", filetypes=["*.snap.parquet"])
print(f'Seleccionado: {ruta_parquet}')

## 
contador_ejecuciones = 0
contador_vueltas_constantes = 0
ls_len_data = []
ls_len_data.append(cantidad_scrap)
while True:
    print('='*120)
    print('='*120)    
    contador_ejecuciones += 1
    # Leer los documentos ya extraídos
    if os.path.exists('BUSQUEDA'):
        ls_archivos_extraidos = os.listdir(path='BUSQUEDA')
        if len(ls_archivos_extraidos) > 0:
            data_buscada0 =pd.DataFrame()
            for i in ls_archivos_extraidos: 
                temp = pd.read_parquet(f'BUSQUEDA/{i}')
                temp['NOMBRE'] = i
                data_buscada0 = pd.concat([data_buscada0,temp],axis = 0)
            data_buscada0.sort_values(by=['nro_documento','NOMBRE'],ascending=True,inplace=True)
            data_buscada0.drop_duplicates(subset='nro_documento',keep='last',inplace=True)

            # extraccion de dnis
            ls_documentos_extraidos = []
            for rowi, rowi2 in data_buscada0[['data','nro_documento']].values:
                try:
                    if 'no se encontraron resultados' in rowi.lower():
                        ls_documentos_extraidos.append(rowi2)
                        continue
                    else:
                        coincidencia_doc = None
                        coincidencia_doc = re.search(string=rowi,pattern=r'\n(.*?)\t')
                        coincidencia_doc = coincidencia_doc.group(1)
                        coincidencia_doc = coincidencia_doc.split(',')[0].strip()
                        coincidencia_doc = coincidencia_doc.split(' ')[-1]
                        ls_documentos_extraidos.append(coincidencia_doc)
                except:
                    ls_documentos_extraidos.append('')

            data_buscada0['documentos_buscado'] = ls_documentos_extraidos #[re.findall(string=rowi,pattern=r'(\d{8,})') for rowi in data_buscada0['data']]
            data_buscada0 = data_buscada0[data_buscada0['nro_documento']==data_buscada0['documentos_buscado']]

    df_all_data = pd.read_parquet(ruta_parquet)
    df_all_data.rename(columns={'NUMDOC':'nro_documento'},inplace=True)
    data_buscada1 = data_buscada0.merge(df_all_data,on='nro_documento')
    data_buscada2 = data_buscada1[data_buscada1['BLOQUE']==bloque_busqueda]
    ls_len_data.append(len(data_buscada2))
    print(f'Datos extraídos al momento: {len(data_buscada2)}')

    if cantidad_scrap <= len(data_buscada2):
        print('Se alcanzó la cantidad buscada')
        break

    if contador_vueltas_constantes > 2:
        print('No se detecta aumento en la cantidad scrapeada')
        break

    cantidad_a_buscar = cantidad_scrap-len(data_buscada2)

    if cantidad_a_buscar < 30:
        numero_procesadores = 1
    comando = f'"{path_python}" "{ruta_script}/SUNEDU.py" {cantidad_a_buscar} {bloque_busqueda} {numero_procesadores} {ruta_parquet}'
    # Ejecutar el comando en CMD
    resultado = subprocess.run(comando, shell=True)

    if ls_len_data[-1] == ls_len_data[-2]:
        contador_vueltas_constantes += 1

    print(f'Termina ejecución {contador_ejecuciones}')
    



#%%

