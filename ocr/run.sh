#!/bin/bash
#export
nohup paddlex --serve \
--pipeline /work/project/paddleOcr/config.yaml \
--port 8206 \
--device "gpu:0" \
>/work/project/paddleOcr/log.txt 2>&1 &