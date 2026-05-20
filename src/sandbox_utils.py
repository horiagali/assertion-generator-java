import json
import os
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


def cleanup_star_files(repo_path):

    for root, dirs, files in os.walk(repo_path):

        for file in files:

            if (
                "STAR" in file
                and file.endswith(".java")
            ):

                try:

                    os.remove(
                        os.path.join(root, file)
                    )

                except Exception:

                    pass