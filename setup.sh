conda create -n astormer python=3.8
source activate astormer
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -c "import stanza; stanza.download('en')"
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
mkdir -p pretrained_models && cd pretrained_models
git lfs install
git clone https://huggingface.co/bert-large-uncased-whole-word-masking
git clone https://huggingface.co/google/electra-large-discriminator
