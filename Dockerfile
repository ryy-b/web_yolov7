FROM python:3.7.13

RUN mkdir /code
WORKDIR /code
COPY yolov7/requirements.txt /code/

RUN pip install -r requirements.txt
COPY . /code/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

RUN pip install fvcore
RUN pip install omegaconf
RUN pip install timm
RUN pip install torchvision==0.10.0
RUN pip install torch==1.9.0
RUN pip install -U jupyter ipykernel

RUN pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI
RUN pip install cloudpickle
RUN pip install streamlit
RUN pip install memory-profiler

# Detec
RUN pip install opencv-python
RUN pip install mss
RUN pip install timm
RUN pip install dataclasses
RUN pip install ftfy
RUN pip install regex
RUN pip install fasttext
RUN pip install scikit-learn
RUN pip install lvis
RUN pip install nltk
RUN pip install git+https://github.com/openai/CLIP.git