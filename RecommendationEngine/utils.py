import os, json
def get_ngrok_url():
    os.system("curl  http://localhost:9000/api/tunnels > tunnels.json")

    with open('tunnels.json') as data_file:    
        datajson = json.load(data_file)

    url = datajson['tunnels'][0]['public_url']
    os.system('rm -r tunnels.json')
    return url