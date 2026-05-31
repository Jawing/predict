import requests
number_of_events = 1000
response = requests.get(f"https://gamma-api.polymarket.com/events?closed=false&limit={number_of_events}")
events = response.json()

all_tags = set()
for event in events:
    tags = event.get('tags', [])
    for tag in tags:
        if isinstance(tag, dict) and 'label' in tag:
            all_tags.add(tag['label'])

print(sorted(list(all_tags)))