# fashion-mnist-tensorflow

基于tensorflow构建的Fashion-MNIST数据集训练DEMO

# 模型训练

## 1、安装依赖包

### MACOS

```bash
pip3 install -r requirements-mac.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Linux

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2、执行训练

```bash
python3 train.py
```

训练完成后在`saved_model`目录中产出模型

# 模型推理

```bash
# 方式一：本地推理
python3 test.py

# 方式二：本地通过HTTP Server推理
# 1、启动HTTP Server推理服务
nohup python3 app.py > run.log 2>&1 &

# 2、发送请求进行推理（以下4种方式可参考）

# 发送base64图片编码推理(python版本)
python3 test_base64_request.py

# 发送base64图片编码推理(shell版本)
sh test_base64_request.sh

# 发送图片文件推理(python版本)
python3 test_file_request.py

# 发送图片文件推理(shell版本)
sh test_file_request.sh
```