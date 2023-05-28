FROM python:3.10.6

WORKDIR /code

RUN apt-get update && \
    apt-get install ffmpeg -y

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/app

RUN ["chmod", "+x", "/code/app/scripts/run_app.sh"]