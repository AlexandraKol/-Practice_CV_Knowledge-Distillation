FROM tensorflow/tensorflow:2.10.0-gpu
RUN apt-get update -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
COPY . .
WORKDIR .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "./train.py" ]