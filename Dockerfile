FROM --platform=linux/aarch64 python:3.8

RUN apt-get -y update

RUN apt install build-essential -y

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get install -y \
        gcc \
        build-essential \
        zlib1g-dev \
        wget \
        unzip \
        cmake \
        python3-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
    && apt-get clean

# Install Python packages
RUN pip install --upgrade pip \
    && pip install \
        ipython[all] \
        numpy \
        nose \
        matplotlib \
        pandas \
        scipy \
        sympy \
        cython \
    && rm -fr /root/.cache

RUN pip install tabgan

RUN pip install table_evaluator

RUN pip install tensorflow

COPY . /

WORKDIR .

RUN python3 -m pip install --upgrade pip

CMD python3 -m data_generator