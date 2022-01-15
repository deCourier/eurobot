#!/usr/bin/env bash

FROM sixsq/opencv-python:master-arm

WORKDIR /tmp
COPY requirements.txt .
COPY multi_thread_example.py .
COPY picamerathread.py .

RUN apt-get -q update
RUN apt-get -q update && apt-get --yes install python-pip && pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python3", "./multi_thread_example.py", "-t 60", "--screen_output"]