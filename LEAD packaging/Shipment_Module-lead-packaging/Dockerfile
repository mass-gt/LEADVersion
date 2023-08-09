FROM python:3.8.12-slim-bullseye

WORKDIR /srv/shipmentsynthesis

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
       liblapack-dev libatlas-base-dev && \
   rm -rf /var/lib/apt/lists/*

COPY setup.py requirements.txt README.md ./
COPY src src

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

ENTRYPOINT [ "shipment-synthesis", "-vv" ]
