import requests
import time
import math
import os, json

from flask import Flask, request, json
from pyngrok import ngrok
from datetime import datetime
from configparser import ConfigParser
from datetime import timedelta
from recommendation import Recommendator
from utils import get_ngrok_url

# Lectura del fichero de configuración (Contiene el token)
config = ConfigParser()
config.read("telegram_config.ini")
telegram_token = config.get('mytokens', 'telegram')

# Creación del tunel seguro desde localhost
os.system('(./ngrok http 8000 &); exit')
time.sleep(2)

# Obtener la URL en la que está publicando el puerto 8000 ngrok
url = get_ngrok_url()
url = url.replace('https://', '')
url = url.replace('http://', '')
print(url)


#Creación del webhook para que Telegram envíe los mensajes
deleteWebHook = 'https://api.telegram.org/bot' + telegram_token + '/deleteWebHook'
response_d = requests.get(deleteWebHook)
setWebHook = 'https://api.telegram.org/bot' + telegram_token + '/setWebHook?url=' + url+'/telegram/'
response = requests.get(setWebHook)
print(setWebHook)
print(response)
print(os.getcwd())

#Sistema de recomendación
recommendator = Recommendator()


app = Flask(__name__)
starting = math.floor(time.time())

@app.route('/')
def welcome():
    return 'Hola mundo'
@app.route('/telegram/', methods = ['POST', 'GET'])
def main ():
    
    # Mensaje recibido a través de TElegram y extracción de parámetros importantes
    recibido = dict(request.json)
    tiempo = recibido['message']['date']
    resta = tiempo-starting
    fecha_bien = datetime.fromtimestamp(starting).strftime("%A, %B %d, %Y %I:%M:%S")
    print(f'App started on {fecha_bien}')
    print(f"Message received {resta} seconds after starting.")
    if recibido["message"]["date"] > starting:
        # Caso en el que el mensaje se haya recibido con la aplicación operativa
        #(si no acumularía todos los mensajes y llegarían de repente)
        chat_id = str(recibido["message"]["chat"]['id'])
        usuario = recibido["message"]["chat"]["first_name"]
        mensaje = str(recibido["message"]["text"])
        print(usuario)
        print(recibido["message"]["text"])
        if mensaje != '/start':
            recommendation = recommendator.predict_from_string(mensaje)
            answer = recommendator.parse_response(recommendation)
        else:
            answer = """Welcome to the Film Recommendation Bot (built by Gonzalo Izaguirre).\n
            Just send us a title and we'll give you similar films."""
        
        send_text = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=Markdown&text=' + answer
        requests.get(send_text)
    return 'Hola Mundo'
if __name__ =='__main__':
    app.run(port = 8000)