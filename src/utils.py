import json

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def append_to_jsonl(path, data):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def get_dataset_size(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)