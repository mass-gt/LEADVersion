FROM python:3.8.12-slim-bullseye

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
       liblapack-dev libatlas-base-dev && \
   rm -rf /var/lib/apt/lists/*

COPY setup.py requirements.txt README.md /srv/tour2network/

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install --no-cache-dir -r /srv/tour2network/requirements.txt

COPY src /srv/tour2network/src
RUN pip install --no-cache-dir /srv/tour2network/

ENTRYPOINT [ "tour2network", "-vv" ]
