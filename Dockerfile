FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
COPY requirements.txt /workspace
ENV NLTK_DATA=/root/nltk_data STANZA_RESOURCES_DIR=/root/stanza_resources
RUN pip3 install -r requirements.txt \
    && python3 -c "import stanza; stanza.download('en')" \
    && python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" \
    && rm -rf $NLTK_DATA/corpora/stopwords.zip $STANZA_RESOURCES_DIR/en/default.zip