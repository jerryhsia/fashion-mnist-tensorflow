import json
import requests
import os

url = "http://127.0.0.1:8886/predict"

for i in range(0, 10, 1):
    dir = 'test/' + str(i)
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        # 打开文件并读取内容
        with open(img_path, 'rb') as file:
            files = {'image': file}

            response = requests.post(url, files=files)
            result = json.loads(response.content)
            print(img_path)
            print(result)
            print('----------')