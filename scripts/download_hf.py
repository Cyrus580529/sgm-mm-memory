"""Download every file of a Hugging Face dataset repo (optionally under a prefix)."""
import json, os, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

repo, out, prefix = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
tree_url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{prefix}?recursive=true"
files = [x["path"] for x in json.load(urllib.request.urlopen(tree_url)) if x["type"] == "file"]

def get(path):
    dst = os.path.join(out, path)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/" + urllib.parse.quote(path)
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, dst + ".part"); os.replace(dst + ".part", dst); return 0
        except Exception as e:
            err = e
    print("FAIL", path, err); return 1

with ThreadPoolExecutor(16) as ex:
    fails = sum(ex.map(get, files))
print(f"{len(files)} files, {fails} failed")
