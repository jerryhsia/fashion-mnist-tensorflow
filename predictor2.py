# -*- coding: utf-8 -*-
"""
用于百度AI中台模型包使用
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import base64


class CustomException(RuntimeError):
    """
    进行模型验证和部署服务必需的异常类，缺少该类在代码验证时将会失败
    在处理异常数据或者请求时，推荐在`PredictWrapper`中的自定义预处理preprocess和后处理postprocess函数中抛出`CustomException`类，
    并为`message`指定准确可读的错误信息，以便在服务响应包中的`error_msg`参数中返回。
    """

    def __init__(self, error_code, message, orig_error=None):
        """ init with error_code, message and origin exception """
        super(CustomException, self).__init__(message)
        self.error_code = error_code
        self.orig_error = orig_error


class PredictWrapper(object):
    """ 模型服务预测封装类，支持用户自定义对服务请求数据的预处理和模型预测结果的后处理函数 """

    def __init__(self, model_path, use_gpu, logger):
        """
        根据`model_path`初始化`PredictWrapper`类，如解析label_list.txt，加载模型输出标签id和标签名称的映射关系
        :param model_path: 该目录下存放了用户选择的模型版本中包含的所有文件
        """
        # 加载模型
        self.loaded_model = tf.saved_model.load(model_path)
        self.inference_func = self.loaded_model.signatures["serving_default"]

        # 定义标签
        self.labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def preprocess(self, request_body):
        """
        自定义对请求体的预处理，针对图像类模型服务，包括对图片对图像的解析、转化等
        :param request_body: 请求体的json字典
        :return:
            data: 用于模型预测的输入。
            args: 用于模型预测的其他参数
            request_context: 透传给自定义后处理函数`postprocess`的参数，例如指定返回预测结果的top N，过滤低score的阈值threshold.
        """
        # 预处理示例：期望请求结构为 {"image": "base64", "top_num": n}
        return request_body, {}, {}

    def predict(self, request_body, args):
        """ 自定义模型预测
        :param request_body: 用于模型预测的输入
        :param args: 用于模型预测的其他参数
        :return: result: 预测结果
        """

        # 图像加载
        image_b64 = request_body['image']
        print(image_b64)
        image_bin = base64.b64decode(image_b64)
        img = image.load_img(BytesIO(image_bin), target_size=(28, 28), color_mode="grayscale")

        # 图像预处理
        x = image.img_to_array(img)
        input_array = x.reshape((1,) + x.shape)
        input_array /= 255

        # 执行模型推理
        input_tensor = tf.convert_to_tensor(input_array)
        output_tensor = self.inference_func(input_tensor)
        predictions = output_tensor['dense_1'].numpy()

        prediction = tf.argmax(predictions, axis=1)
        index = prediction.numpy()[0]
        label = self.labels[index]

        result = {"index": str(index), "label": label}
        return result

    def postprocess(self, infer_result, request_context):
        """
        对模型预测结果进行后处理
        :param infer_result: 模型的预测结果
        :param request_context: 自定义预处理函数中返回的`request context`
        :return: request results 请求的处理结果
        """
        # 对预测结果进行后处理，封装请求返回结果
        # case 1: 当为实现自定义`predict`方法时，预测逻辑由ModelServer内置函数完成，预测结果结构可参考ModelServer标准的请求返回结果（未实现自定义逻辑时）
        # output_tensor_info = infer_result['result']['predictions'][0]["tensors"][0]
        # output_tensor_data = np.array(output_tensor_info["data"], dtype="float32")

        # case2：当实现了自定义`predict`方法时，`infer_result`即为`predict`方法的返回结果。
        return infer_result
