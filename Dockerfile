from tensorflow/tensorflow:2.6.0-gpu

WORKDIR /root/fashion

COPY *.py *.sh /root/fashion/
COPY test /root/fashion/test

RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install Pillow flask requests gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 cache purge

RUN python3 preload.py