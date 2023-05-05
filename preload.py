from tensorflow.keras.datasets import fashion_mnist
import os
import ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# 加载数据集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
