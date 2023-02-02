conda create -n text2sql python=3.7
source activate text2sql
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -c "import stanza; stanza.download('en')"
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
mkdir -p pretrained_models && cd pretrained_models
git lfs install
git clone https://huggingface.co/bert-large-uncased-whole-word-masking
git clone https://huggingface.co/google/electra-large-discriminator