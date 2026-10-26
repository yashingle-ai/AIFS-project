import json

path = "storage/Earnings_Presentation-16935aea06ca/chunks.json"
chunks = json.loads(open(path, encoding="utf-8").read())

img_chunks = [c for c in chunks if c.get("kind") in ("image", "table") and c.get("image_file")]
print(f"Total chunks: {len(chunks)}")
print(f"Image/chart chunks (with image_file): {len(img_chunks)}")
print(f"Pure image chunks: {sum(1 for c in chunks if c.get('kind') == 'image')}")
print(f"Table-from-chart chunks: {sum(1 for c in img_chunks if c.get('kind') == 'table')}")
print()

for c in img_chunks[:8]:
    print(f"--- chunk_id={c['chunk_id']}, page={c['page']}, kind={c['kind']}, image_file={c.get('image_file')}")
    print(c["text"][:400])
    print()
