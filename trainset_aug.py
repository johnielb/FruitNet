import os
import requests

blacklist_cherry = ["blossom", "berry", "strawberry", "tomato", "flower"]
blacklist_strawberry = ["cherry", "dessert", "ice cream", "tomato", "flower", "drink", "breakfast"]
blacklist_tomato = ["sauce", "vegetables", "strawberry", "cherry", "flower", "animal", "onion", "spaghetti", "egg",
                    "bread", "sandwich", "salad", "food", "soup", "bacon"]

cherry_urls = []
page = 1
while len(cherry_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "cherry",
              "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_cherry): continue
        cherry_urls.append(hit["largeImageURL"])
    page += 1

strawberry_urls = []
page = 1
while len(strawberry_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "strawberry",
              "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_strawberry): continue
        strawberry_urls.append(hit["largeImageURL"])
    page += 1

tomato_urls = []
page = 1
while len(tomato_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "tomato",
              "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_tomato): continue
        tomato_urls.append(hit["largeImageURL"])
    page += 1

base_dir = "traindata"
for i, url in enumerate(cherry_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "cherry", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)
for i, url in enumerate(strawberry_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "strawberry", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)
for i, url in enumerate(tomato_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "tomato", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)