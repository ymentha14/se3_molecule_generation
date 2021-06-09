FROM python:3.6.9
COPY ./ ./app
WORKDIR ./app

RUN apt-get update && \
    apt-get -y install curl && \
    apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates && \
    curl -sL https://deb.nodesource.com/setup_12.x | bash && \
    apt-get -y install nodejs

RUN apt-get -y install make


# Install dependencies:
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
CMD [ "/bin/bash"]


