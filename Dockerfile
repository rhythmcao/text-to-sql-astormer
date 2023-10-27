ARG PYTORCH_VERSION="1.8.1"
ARG CUDA_VERSION="11.1"
ARG CUDNN_VERSION="8"
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

COPY ./requirements.txt /workspace
COPY ./stanza_resources /root/stanza_resources
COPY ./nltk_data /root/nltk_data
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -c "import jionlp" \
    && sed -i "s/level='INFO', log_dir_name='\.jionlp_logs'/level='ERROR', log_dir_name=None/" /opt/conda/lib/python3.8/site-packages/jionlp/__init__.py \
    && rm -rf /root/.cache/pip/*
