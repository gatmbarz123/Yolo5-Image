import json
import time
from decimal import Decimal
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import requests

aws_region  =   os.getenv("AWS_REGION")
dynamodb_table=os.getenv("DYNAMODB_TABLE")
alb_url =   os.getenv("ALB_URL")
s3_bucket   =   os.getenv("S3_BUCKET")
sqs_url =   os.getenv("SQS_URL")

images_bucket = s3_bucket
queue_name = sqs_url

sqs_client = boto3.client('sqs', region_name= aws_region)
s3 = boto3.client('s3')
dynamodb_clinet = boto3.resource('dynamodb', region_name= aws_region)

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def consume():
    while True:

        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)
        if 'Messages' in response:
            message = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')
            message_dict = json.loads(message)

            # Receives a URL parameter representing the image to download from S3
            img_name = message_dict["image_name"] 
            chat_id = message_dict["chat_id"]
            logger.info(f'Message received: {message_dict["image_name"]}') 

            local_img_dir = 'tempImages'
            os.makedirs(f'{local_img_dir}/{chat_id}', exist_ok=True)
            original_img_path = os.path.join(local_img_dir, img_name)
            
            s3.download_file(images_bucket,img_name,original_img_path)
            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
            
            
            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels
            # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
            predicted_img_path = Path(f'static/data/{prediction_id}/{img_name.split("/")[-1]}')

            # Upload the predicted image to S3
            predicted_img_s3_path = f'predictions/{prediction_id}/{img_name}'

            # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
            s3.upload_file(predicted_img_path, images_bucket, f'{img_name}.prediction')

            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split("/")[-1][:-4]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': Decimal(str(l[1])),
                        'cy': Decimal(str(l[2])),
                        'width': Decimal(str(l[3])),
                        'height': Decimal(str(l[4])),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

                prediction_summary = {
                    'prediction_id': prediction_id,
                    'original_img_path': str(original_img_path),
                    'predicted_img_path': str(predicted_img_path),
                    'labels': labels,
                    'time': Decimal(str(time.time())),
                    'chat_id': chat_id
                }

                # TODO store the prediction_summary in a DynamoDB table
                table = dynamodb_clinet.Table(dynamodb_table)
                try:
                    table.put_item(Item=prediction_summary)
                    logger.info(f'It Work!! Successfully stored prediction summary for {prediction_id}')
                except Exception as e:
                    logger.error(f'Error storing prediction summary in DynamoDB: {e}')
                

                # TODO perform a GET request to Polybot to `/results` endpoint
                loadbalancer_domain = f'https://{alb_url}:8443'
                try:
                    response = requests.post(f'{loadbalancer_domain}/results', params={'prediction_id': prediction_id})
                    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                    logger.info(f'Notified Polybot about results for prediction_id: {prediction_id}')
                except requests.exceptions.RequestException as e:
                    logger.error(f'Failed to notify Polybot: {e}')

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
