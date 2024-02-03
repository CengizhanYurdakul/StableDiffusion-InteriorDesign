FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS BASE

ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app
COPY . /app/

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r /app/requirements.txt

ENTRYPOINT ["python", "/app/app.py"]