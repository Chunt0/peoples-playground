#!/bin/bash

# Set the bucket path and local destination path
BUCKET_PATH="gs://peoples-shared/test/"
LOCAL_DEST="./"

# Copy files from the bucket to the local destination
gcloud storage cp -r $BUCKET_PATH $LOCAL_DEST

# Iterate through the downloaded directory and print each file
echo "Downloaded files:"

for file in "$LOCAL_DEST"*
do 
    if [ -f "$file" ]; then
        echo "File: $file"
    fi
done