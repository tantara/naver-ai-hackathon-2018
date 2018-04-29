#!/bin/bash

NAVER_AI_PATH=`pwd`

nvidia-docker run -it -v $NAVER_AI_PATH:/naver-ai \
  --name tantara-naver-ai -d tantara-naver-ai
