FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y \
    python3-pip \
    vim && \
    pip3 install scipy

WORKDIR /code/ 

CMD ["bash","run_deep.sh","37"]

