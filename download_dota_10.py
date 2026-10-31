import os
import random
from io import BytesIO

import requests
from PIL import Image

DATASET = "finedet/dota"
CONFIG = "v1.0"
SPLIT = "train"
TARGET = 10

CLASSES = [
    "plane", "ship", "storage-tank",
    "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field",
    "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter",
    "roundabout", "soccer-ball-field",
    "swimming-pool"
]

os.makedirs("images", exist_ok=True)
os.makedirs("labelTxt", exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "DOTA-research-downloader/1.0"})

random.seed(42)

# Randomly visit 100-row pages within the 1411-image train split.
offsets = list(range(0, 1411, 100))
random.shuffle(offsets)

count = 0

for offset in offsets:
    try:
        response = session.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": min(100, 1411 - offset)
            },
            timeout=90
        )
        response.raise_for_status()
        rows = response.json()["rows"]
    except Exception as e:
        print("Page error at offset {}: {}".format(offset, e))
        continue

    random.shuffle(rows)

    for item in rows:
        row = item["row"]
        objects = row["objects"]

        categories = objects["category"]

        # Only select images containing vehicles.
        if not any(int(c) in (9, 10) for c in categories):
            continue

        image_info = row["image"]
        image_url = image_info.get("src")

        if not image_url:
            print("Missing image URL, skipping")
            continue

        name = "dota_train_{:04d}".format(item["row_idx"])

        try:
            image_response = session.get(image_url, timeout=120)
            image_response.raise_for_status()

            image = Image.open(BytesIO(image_response.content))
            image = image.convert("RGB")

            lines = []

            for polygon, category, difficult in zip(
                objects["polygon"],
                categories,
                objects["difficult"]
            ):
                if isinstance(polygon[0], (list, tuple)):
                    polygon = [v for pt in polygon for v in pt]

                if len(polygon) != 8:
                    continue

                class_name = CLASSES[int(category)]

                coordinates = " ".join(
                    str(int(round(float(v)))) for v in polygon
                )

                lines.append(
                    "{} {} {}".format(
                        coordinates,
                        class_name,
                        int(difficult)
                    )
                )

            image.save("images/" + name + ".jpg", quality=95)

            with open("labelTxt/" + name + ".txt", "w") as f:
                f.write("\n".join(lines) + "\n")

            count += 1
            print("Downloaded {}/{}: {}".format(
                count, TARGET, name
            ), flush=True)

        except Exception as e:
            print("Skipping {}: {}".format(name, e))

        if count >= TARGET:
            break

    if count >= TARGET:
        break

print("\nCompleted: {} images".format(count))
