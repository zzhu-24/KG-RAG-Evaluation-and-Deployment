import json

file_path = './data/hotpot_dev_distractor_v1.json'
data = json.load(open(file=file_path))

print('number of question : ', len(data))


titles = set()
for ex in data:
    for title, _ in ex["context"]:
        titles.add(title)

print("The number of documents: ", len(titles))


with open("processed/documents.jsonl") as f:
    for i in range(3):
        print(json.loads(next(f))["text"])
        print("-----")
