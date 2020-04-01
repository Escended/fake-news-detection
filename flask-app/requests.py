import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'article'})

print(r.json())