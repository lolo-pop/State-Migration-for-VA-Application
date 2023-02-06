# FROM jamshi/python3.7-torch-cuda:latest

FROM nvcr.io/nvidia/tensorflow:21.12-tf1-py3

# RUN conda install -y tensorflow-gpu=1.15

RUN pip install scipy \ 
                sklearn \ 
                opencv-python \ 
                matplotlib \ 
                pycocotools \ 
                psutil \ 
                pyaml \ 
                redis \ 
                flask \ 
                flask-cors

RUN pip install dapr \ 
                dapr-ext-grpc \ 
                dapr-ext-fastapi

EXPOSE 5000

RUN apt update --fix-missing
RUN apt install -y --no-install-recommends libgl1-mesa-glx

RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY . .

