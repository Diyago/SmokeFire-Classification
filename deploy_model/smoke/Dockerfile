FROM ubuntu:19.10

COPY ./models /models
COPY requirements.txt /requirements.txt
COPY main.py /main.py
COPY test_image_label0.png /test_image.png
COPY test_image_label1.png /test_image.png

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

ENV PYTHONPATH=/api
WORKDIR /

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]