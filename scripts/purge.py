"""Delete stored evaluation data after a run (official data may not be kept or reused).

Removes the database (memories, descriptions, embedding cache) and all stored images under MEMORY_DATA_DIR.
Stop the service first. Asks for confirmation unless --yes is given.

usage: (set -a; . ./.env; set +a; python scripts/purge.py [--yes])
"""
import os
import shutil
import sys

data_dir = os.path.abspath(os.environ.get("MEMORY_DATA_DIR", "artifacts"))
targets = [os.path.join(data_dir, n) for n in ("memory.db", "memory.db-wal", "memory.db-shm", "images")]
present = [t for t in targets if os.path.exists(t)]
if not present:
    sys.exit(f"nothing to delete under {data_dir}")
size = sum(os.path.getsize(os.path.join(r, f)) for t in present if os.path.isdir(t)
           for r, _, fs in os.walk(t) for f in fs) + sum(os.path.getsize(t) for t in present if os.path.isfile(t))
print(f"will delete under {data_dir} ({size / 1e9:.2f} GB):\n  " + "\n  ".join(present))
if "--yes" not in sys.argv and input("type 'delete' to confirm: ").strip() != "delete":
    sys.exit("aborted")
for t in present:
    shutil.rmtree(t) if os.path.isdir(t) else os.remove(t)
print("deleted")
