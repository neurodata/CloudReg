cd bias_correction_lambda/lib/python3.8/site-packages
zip -r9 ~/MBAC/preprocessing/lambda_function.zip .
cd ~/MBAC/preprocessing/
zip -g lambda_function.zip lambda_correct_tile.py
aws lambda update-function-code --function-name colm_tile_correction --zip-file fileb://lambda_function.zip
