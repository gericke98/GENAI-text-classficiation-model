import json
import boto3
from dotenv import load_dotenv
import os

def lambda_handler(event, context):
    load_dotenv()

    sagemaker_runtime = boto3.client('sagemaker-runtime')

    body = json.loads(event['body'])

    headline = body['query']['headline']

    #headline = "How I met your Mother voted as best sitcom in Europe" # Test headline

    endpoint_name = os.getenv('MODEL_ENDPOINT')

    payload = json.dumps({"inputs": headline})

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = "application/json",
        Body = payload
        )
    result = json.loads(response['Body'].read().decode())

    return {

        'statusCode':200,
        'body': json.dumps(result)
    }