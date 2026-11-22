import urllib.request
import time
import os

OUTPUT_FILE = "github_train_data.txt"

# Direct links to massive, high-quality Python standard library files on GitHub
RAW_FILES = [
    "https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/datetime.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/decimal.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/inspect.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/tarfile.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/pydoc.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/logging/__init__.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/concurrent/futures/thread.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/socket.py",
    "https://raw.githubusercontent.com/django/django/main/django/db/models/query.py",
    "https://raw.githubusercontent.com/django/django/main/django/core/handlers/base.py"
]

print("Downloading massive Python codebase from GitHub...")
collected_text = []
total_chars = 0

for url in RAW_FILES:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
            collected_text.append(content)
            total_chars += len(content)
            print(f"Fetched: {url.split('/')[-1]:<20} | Size: {len(content):,} characters")
            
            # Brief pause to respect server requests
            time.sleep(0.5)
            
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

# Save all scraped code into your training file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n# ==========================================\n\n".join(collected_text))

print(f"\nSuccess! All files merged into '{OUTPUT_FILE}'.")
print(f"Total Dataset Size: {total_chars:,} characters (~{total_chars / 1024 / 1024:.2f} MB)")