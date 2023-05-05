import json
import requests
import os
import base64

url = "http://127.0.0.1:8886/predict"
headers = {"Content-Type": "application/json"}

for i in range(0, 10, 1):
    dir = 'test/' + str(i)
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        # 打开文件并读取内容
        with open(img_path, 'rb') as file:
            file_content = file.read()

        # 转换为 base64 编码
        encoded_content = str(base64.b64encode(file_content), encoding='utf-8')

        data = json.dumps({"image": encoded_content})

        response = requests.post(url, data, headers=headers)
        result = json.loads(response.content)
        print(img_path)
        print(result)
        print('----------')
