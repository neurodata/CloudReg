#!/bin/bash

mkdir -p ~/.cloudvolume/secrets/
# copy over cloudvolume secret
cp ${CV_CRED_PATH} ~/.cloudvolume/secrets/aws-secret.json

python -m CloudReg.cloudreg.scripts.colm_pipeline $@
