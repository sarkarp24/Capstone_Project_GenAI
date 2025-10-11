import requests
import json

# local API
url = "http://localhost:8005/medical_assistance/"

question = {
    "question": "What is Type 2 Diabetes and what are the treatments available?"
}
'''
question = {
    "question": "What is capital of India?"
}
'''
# Send POST request
response = requests.post(url, json=question)

# Print response
if response.status_code == 200:
    print((response.json()['answer']))
else:
    print(response.status_code, response.text, response.reason,response,response.content)