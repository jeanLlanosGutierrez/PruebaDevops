#%%
from PIL import Image
from io import BytesIO
import os

import easyocr
import time
import re
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright
import asyncio
import threading
import datetime
import signal
import sys
import warnings
import easygui

warnings.filterwarnings("ignore")

# Para obtener la ruta del directorio donde se encuentra el script
ruta_script = os.path.dirname(os.path.abspath(__file__))

#ruta_script = os.path.dirname(os.path.abspath(sys.executable))
os.chdir(ruta_script)


def resource_path_base(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    # return os.path.join(base_path, relative_path)
    return base_path

modelo_reconocimiento = 'craft_mlt_25k.pth'
path_reconocimiento = resource_path_base(modelo_reconocimiento)

def dividir_lista(lista, n):
    # Calcular el tamaño de cada sublista
    tamaño_sublista = len(lista) // n
    # Crear las sublistas
    sublistas = [lista[i * tamaño_sublista: (i + 1) * tamaño_sublista] for i in range(n)]
    # Añadir los elementos restantes a la última sublista
    sublistas[-1].extend(lista[n * tamaño_sublista:])
    return sublistas


def save_dataframe(ls_negativos):
    if len(ls_negativos[0]) > 0:
        data_frame_general = pd.concat([pd.DataFrame(i,columns=['nro_documento','data']) for i in ls_negativos],axis=0)
        fecha_hoy = datetime.datetime.today()
        fecha_hora_corrida_str = fecha_hoy.strftime(format='%Y-%m-%d %H%M')
        # Verifica si el directorio existe, si no, lo crea
        if not os.path.exists('BUSQUEDA'):
            os.makedirs('BUSQUEDA')
        data_frame_general.to_parquet(f'BUSQUEDA/file {fecha_hora_corrida_str} - {os.getlogin()}.snap.parquet',compression='snappy',index=False)
    else:
        print('Nada para guardar.')

def signal_handler(sig, frame):
    print('Interrupción recibida, ...')
    save_dataframe(ls_negativos)
    print('Saliendo del programa.')
    sys.exit(0)


#%%
async def run(ls_dnis,ls_negativos,reader,num_procesador):

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://constancias.sunedu.gob.pe/verificainscrito")
        # Deshabilitar el desplazamiento con JavaScript
        await page.wait_for_load_state('networkidle')

        contador_avance = 0
        for dni in ls_dnis: 
            dni = dni.strip()

            ## REINICIAR LAS VARIABLES
            dni_campo = None
            captcha_locator = None
            captcha_bytes = None
            image = None
            resultado_imagen = None
            captcha_text = None
            campo_captcha = None
            boton_cerrar_error = None
            error = None
            error_msg = None
            resultado = None

            try:
                indicador = 0

                try:
                    await page.wait_for_selector('.btn.btn-default', state='visible',timeout=1500)
                    await page.click('.btn.btn-default',timeout=1500)
                    await page.click('#VerirefrescarCaptcha')
                    #print('Se refresca el capcha')
                    await page.wait_for_timeout(1600)
                except:
                    await browser.close()
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto("https://constancias.sunedu.gob.pe/verificainscrito")
                    await page.wait_for_load_state('networkidle')
                    #print('Se abre otro navegador')

                # Seleccionar el campo por XPath y rellenarlo con el valor
                dni_campo = page.locator('//*[@id="doc"]')  # Seleccionamos el campo por su XPath
                await dni_campo.fill(dni)  # Enviar el DNI

                # Justo antes de capturar el captcha
                try:
                    captcha_locator = page.locator('//*[@id="captchaImg"]/img')
                    # Esperamos que el captcha esté visible y unido al DOM
                    await captcha_locator.wait_for(state='visible', timeout=2000)
                    # Capturamos el screenshot directamente
                    captcha_bytes = await captcha_locator.screenshot(type='png')

                except Exception as e:
                    #print(f"[{dni}] Error al capturar captcha: {e}")
                    ls_negativos.append({'nro_documento': dni, 'data': 'ERROR: Otro'})
                    await page.reload(wait_until="networkidle", timeout=4000)
                    continue


                # Convertir los bytes en una imagen utilizando PIL
                image = Image.open(BytesIO(captcha_bytes))
                resultado_imagen = reader.readtext(np.array(image))

                # Extraer el texto desde la imagen
                captcha_text = ''
                if resultado_imagen:
                    captcha_text = resultado_imagen[0][1]
                captcha_text = captcha_text.upper()
                #print(captcha_text)

                # Se hace una validación superficial mediante regex, letra mayuscula, numero y de longitud 5
                if re.findall(pattern = r'([A-Z|0-9]{5})', string = captcha_text):
                    pass
                else:
                    ls_negativos.append({'nro_documento':dni, 'data': 'ERROR'})
                    continue

                # Seleccionar el campo donde se debe introducir el captcha
                campo_captcha = page.locator('//*[@id="captcha"]')  # Ajusta el XPath según tu caso
                await campo_captcha.fill(captcha_text)  # Rellenar el campo con el texto del captcha

                # Localizar y hacer clic en el botón de búsqueda
                await page.locator('#buscar').click(timeout=3000)

                error = False
                try:
                    boton_cerrar_error = page.locator('//*[@id="closeModalError"]')
                    # Esperamos que el boton esté visible
                    await boton_cerrar_error.wait_for(state='visible', timeout=1500)
                    error = True 
                except:
                    error = False


                if error:
                    error_msg = await page.inner_text('#frmError_Body',timeout=1000)
                    #print(f"Error: {error_msg}")
                    if "No se encontraron resultados" in error_msg:
                        ls_negativos.append({'nro_documento':dni, 'data':f'ERROR: {error_msg}'})
                        indicador = 0
                    elif "CAPTCHA" in error_msg:
                        ls_negativos.append({'nro_documento':dni, 'data':f'ERROR: {error_msg}'})
                        indicador = 0
                    else:
                        ls_negativos.append({'nro_documento':dni, 'data':f'ERROR: {error_msg}'})
                        indicador = 0
                    
                    await page.click('//*[@id="closeModalError"]') #await page.click('#closeModalError')
                    #print('mensaje error')

                else:
                    await page.wait_for_selector("div.panel-heading:has-text('Resultado')", timeout=3000)
                    await page.inner_html('div.table-responsive-1')
                    resultado = await page.inner_text('#finalData',timeout=3000)
                    if (resultado == '') | ('\t\t\t\t\t' in resultado):
                        resultado = await page.inner_text('#finalDataExt')
                    ls_negativos.append({'nro_documento':dni, 'data':resultado})


            except:
                ls_negativos.append({'nro_documento':dni, 'data':'ERROR: Otro'})
            finally:
                contador_avance += 1 
                if contador_avance%100 == 0:
                    print(f'Avance del procesador {num_procesador}: {contador_avance}')


# Función normal que envuelve la función asíncrona, pero sin usar asyncio.run()
def ejecutar_run(ls_dnis, ls_negativos, reader, num_procesador):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run(ls_dnis, ls_negativos, reader, num_procesador))
    loop.close()

# =============================================================================================================================================
## IMPUTS

print(sys.argv)
cantidad_scrap = int(sys.argv[1])
bloque_busqueda = int(sys.argv[2])
numero_procesadores = int(sys.argv[3])
ruta_parquet = sys.argv[4]

# cantidad_scrap = int(input("Número de registros a buscar:"))
# bloque_busqueda = int(input("Número de bloque a buscar:"))
# numero_procesadores = int(input("Número de procesadores:"))
# ruta_parquet = easygui.fileopenbox(msg="Seleccionar el archivo de base de búsqueda.", filetypes=["*.snap.parquet"])
# print(f'Seleccionado: {ruta_parquet}')




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

        # data_buscada1 = data_buscada0[~(((data_buscada0['data'].str.contains('ERROR'))&(data_buscada0['data'].str.contains('None')))|(data_buscada0['data'].str.contains('CAPTCHA'))|(data_buscada0['data'].str.contains('\t\t\t\t\t'))|(data_buscada0['data']=='')|(data_buscada0['data']=='ERROR')|(data_buscada0['data']=='ERROR: Otro'))];print(f'Extraídos correctamente: {len(data_buscada1)}')
        # data_buscada1['documentos_buscado'] = [re.findall(string=rowi,pattern=r'(\d{8,})') for rowi in data_buscada1['data']]
        # data_buscada1['nro_documento_extraido'] = [rowi2 if ('No se encontraron' in rowi1) else rowi3[0] if len(rowi3)>0 else '' for rowi1, rowi2, rowi3 in data_buscada1[['data','nro_documento','documentos_buscado']].values]
        # data_buscada2 = data_buscada1[data_buscada1['nro_documento']==data_buscada1['nro_documento_extraido']]


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
        # data_buscada0['nro_documento_extraido'] = [rowi2 if ('No se encontraron' in rowi1) else rowi3[0] if len(rowi3)>0 else '' for rowi1, rowi2, rowi3 in data_buscada0[['data','nro_documento','documentos_buscado']].values]
        data_buscada0 = data_buscada0[data_buscada0['nro_documento']==data_buscada0['documentos_buscado']]
        ls_documentos_buscados = data_buscada0['nro_documento'].unique().tolist()
        set_documentos_buscados = set(ls_documentos_buscados)
        #print(f'Numero de documentos extraidos correctamente: {len(ls_documentos_buscados)}')

    else:
        ls_documentos_buscados = []
        set_documentos_buscados = set(ls_documentos_buscados)
else:
    ls_documentos_buscados = []
    set_documentos_buscados = set(ls_documentos_buscados)



## DOCUMENTOS A REALIZAR WEBSCRAPING
df_documentos_sunedu = pd.read_parquet(ruta_parquet)
df_documentos_sunedu = df_documentos_sunedu[df_documentos_sunedu['BLOQUE']==bloque_busqueda]

ls_dni0 = df_documentos_sunedu['NUMDOC'].tolist()
ls_dni0 = [hi for hi in ls_dni0 if hi not in set_documentos_buscados]
ls_dnis = ls_dni0[:cantidad_scrap]

ls_dnis = dividir_lista(ls_dnis, numero_procesadores)

print('='*50)
for element_dni in range(len(ls_dnis)):
    print(f'El procesador {element_dni} tiene {len(ls_dnis[element_dni])} documentos.')

ls_negativos = [[] for _ in range(numero_procesadores)]
ls_reades = [easyocr.Reader(['es'], model_storage_directory=f'{path_reconocimiento}', download_enabled=False) for _ in range(numero_procesadores)]

# Asignar el manejador de señales
signal.signal(signal.SIGINT, signal_handler)

hilos = []
inicio = time.time()
# Crear y empezar varios hilos
for i in range(numero_procesadores):
    hilo = threading.Thread(target=ejecutar_run, args=(ls_dnis[i], ls_negativos[i], ls_reades[i], i,))
    hilos.append(hilo)
    hilo.start()

# Esperar a que todos los hilos terminen
for hilo in hilos:
    hilo.join()

save_dataframe(ls_negativos)
print("Todos los hilos han terminado.")


end = time.time()
print(end - inicio)






