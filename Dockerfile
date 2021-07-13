FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y \
    python3-pip \
    vim && \
    pip3 install scipy boto3 &&\
    mkdir /code

COPY deep_nnEq.py /code

COPY run_deep.sh /code

WORKDIR /code/ 

CMD ["bash","run_deep.sh","37"]

