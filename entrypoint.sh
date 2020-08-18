#! /bin/bash

mkdir -p ~/.cloudvolume/secrets/
# copy over cloudvolume secret
cp ${CV_CRED_PATH} ~/.cloudvolume/secrets/aws-secret.json

python CloudReg/scripts/run_colm_pipeline_ec2.py 