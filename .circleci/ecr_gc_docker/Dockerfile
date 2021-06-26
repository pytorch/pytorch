FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

ADD gc.py /usr/bin/gc.py

ADD docker_hub.py /usr/bin/docker_hub.py

ENTRYPOINT ["/usr/bin/gc.py"]
