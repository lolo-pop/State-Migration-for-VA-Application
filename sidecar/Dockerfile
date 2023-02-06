FROM python:3.7.12

RUN apt update

RUN pip install dapr \ 
                dapr-ext-grpc \ 
                dapr-ext-fastapi \ 
                opencv-python \ 
                flask \ 
                flask-cors \ 
                json-tricks
                

RUN apt install -y --no-install-recommends libgl1-mesa-glx

RUN pip install requests 
RUN pip install kubernetes

COPY . .