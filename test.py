import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

export_dir = 'saved_model/'
loaded_model = tf.saved_model.load(export_dir)
inference_func = loaded_model.signatures["serving_default"]

# 定义标签
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(0, 10, 1):
    dir = 'test/' + str(i)
    for filename in os.listdir(dir):
        # 加载图像
        img_path = os.path.join(dir, filename)
        img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

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

        print(img_path + ' expect:' + str(i) + ' actual:' + str(index) + ' labal:' + label)
