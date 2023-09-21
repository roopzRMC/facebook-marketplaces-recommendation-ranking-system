import json
import requests

response = requests.get("http://127.0.0.1:8000/healthcheck")

print(response.text)

## Test sending an image

test_image_filepath = "ehs-19.jpg"

test_image = open(test_image_filepath, "rb")

payload = {"image": test_image}

image_push = requests.post("http://127.0.0.1:8000/predict/feature_embedding", files = payload)

print(image_push.text)

# Check the response
if response.status_code == 200:
    print("Image uploaded successfully!")
else:
    print("Failed to upload the image.")