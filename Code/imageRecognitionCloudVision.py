import io
import os
from google.cloud import vision
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file("Project_MMME3083\\Code\\food-recognition-441215-d265ab507759.json")
client = vision.ImageAnnotatorClient(
    credentials=creds,
)

# The name of the image file to annotate
file_name = os.path.join( os.path.dirname(__file__), "C:\\Users\\nsant\\OneDrive\\Documents\\Uni\\Y3\\Project_MMME3083\\Code\\apple.jpg")
# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()
request = {
    "image": {"content": content},    
    "features": [
        {"max_results": 2,
        "type": "LABEL_DETECTION"
        },
        {"type": "SAFE_SEARCH_DETECTION"}
    ]
}
response = client.annotate_image(request)
print(response)
print(response.safe_search_annotation.adult)
for label in response.label_annotations:
    print(label.description)