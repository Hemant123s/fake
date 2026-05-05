import requests, json
r = requests.post('http://127.0.0.1:5000/live', json={'query': 'india'})
data = r.json()
arts = data.get('articles', [])
for a in arts:
    print(f"{a['prediction']:5s} | {a['title'][:80]}")
