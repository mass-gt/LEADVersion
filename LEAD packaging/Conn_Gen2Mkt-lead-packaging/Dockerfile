FROM python:3.8.12-slim-bullseye

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
       liblapack-dev libatlas-base-dev && \
   rm -rf /var/lib/apt/lists/*

WORKDIR /srv/gen2mkt

COPY setup.py requirements.txt README.md /srv/gen2mkt/

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install --no-cache-dir -r /srv/gen2mkt/requirements.txt

COPY src /srv/gen2mkt/src
RUN pip install --no-cache-dir /srv/gen2mkt/

WORKDIR /
ENTRYPOINT [ "gen2mkt", "-vv" ]
