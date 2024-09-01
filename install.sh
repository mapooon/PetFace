pip install easydict mxnet onnx scikit-learn timm tensorboard scipy==1.7.3

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
apt-get install ffmpeg libsm6 libxext6  git -y
pip install opencv-python-headless==4.5.5.64 pandas==1.3.5
pip install seaborn matplotlib
pip install git+https://github.com/openai/CLIP.git