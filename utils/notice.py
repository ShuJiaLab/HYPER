import requests

KEY = 'cBbZ7MtxhjuQo23sZZawvo'

def send_notice(text):
    url = "https://maker.ifttt.com/trigger/DL_notify_iOS/with/key/"+KEY+""
    # payload = "{\n    \"value1\": \""+text+"\"\n}"
    payload = "{\"value1\": \""+text+"\"}"
    headers = {
    'Content-Type': "application/json",
    'User-Agent': "PostmanRuntime/7.15.0",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
    'Host': "maker.ifttt.com",
    'accept-encoding': "gzip, deflate",
    'content-length': "63",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
    }
 
    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)
 
    # text = "603609.SH 特大单资金量急剧上增！"
    # send_notice(text)

if __name__ == '__main__':
    send_notice('This is a test!')