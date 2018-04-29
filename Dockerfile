FROM floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17
MAINTAINER Taekmin Kim <tantara.tm@gmail.com>

RUN apt-get update && apt-get install htop screen

ENV APP_PATH /naver-ai
RUN mkdir -p $APP_PATH
WORKDIR $APP_PATH

RUN pip install nltk gensim jpype1 konlpy
RUN pip install git+https://github.com/n-CLAIR/nsml-local.git

ADD . $APP_PATH
