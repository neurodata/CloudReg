#! /bin/bash

mkdir -p ~/.cloudvolume/secrets/
# copy over cloudvolume secret
cp ${CV_CRED_PATH} ~/.cloudvolume/secrets/aws-secret.json

python CloudReg/scripts/colm_pipeline.py $@