import json
from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

app = Flask(__name__)

# 加载模型
export_dir = 'saved_model/'
loaded_model = tf.saved_model.load(export_dir)
inference_func = loaded_model.signatures["serving_default"]

# 定义标签
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 加载图像
def load_image(request):
    if 'image' in request.files:
        # 通过文件上传
        img_file = request.files['image']
        image_data = img_file.read()
    else:
        # 通过base64
        body = request.get_json()
        image_data = base64.b64decode(body['image'])

    img = image.load_img(BytesIO(image_data), target_size=(28, 28), color_mode="grayscale")
    return img


# 定义推理接口
@app.route('/predict', methods=['POST'])
def predict():
    img = load_image(request)

    # 图像预处理
    x = image.img_to_array(img)
    input_array = x.reshape((1,) + x.shape)
    input_array /= 255

    # 执行模型推理
    input_tensor = tf.convert_to_tensor(input_array)
    output_tensor = inference_func(input_tensor)
    predictions = output_tensor['dense_1'].numpy()

    prediction = tf.argmax(predictions, axis=1)
    index = prediction.numpy()[0]
    label = labels[index]

    # 返回预测结果
    return json.dumps({'label': label, 'index': str(index)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8886, debug=True)