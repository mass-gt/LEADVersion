FROM python:3.8-slim-bullseye

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
       liblapack-dev libatlas-base-dev && \
   rm -rf /var/lib/apt/lists/*

COPY setup.py requirements.txt README.md /srv/app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /srv/app/requirements.txt

COPY src /srv/app/src
RUN pip install --no-cache-dir /srv/app/

ENTRYPOINT [ "parcel-tour-formation", "-vvv" ]
