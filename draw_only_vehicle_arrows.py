import os
import argparse
import cv2
import numpy as np


VEHICLE_CLASSES = {"small-vehicle", "large-vehicle"}


def parse_dota_label_file(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue

            pts = np.array(coords, dtype=np.float32).reshape(4, 2)
            class_name = parts[8]

            objects.append({
                "points": pts,
                "class_name": class_name
            })
    return objects


def polygon_center(pts):
    return np.mean(pts, axis=0)


def midpoint(a, b):
    return (a + b) / 2.0


def heading_target_from_p1(pts):
    p1, p2, p3, p4 = pts

    d12 = np.linalg.norm(p2 - p1)
    d41 = np.linalg.norm(p1 - p4)

    if d12 <= d41:
        return midpoint(p1, p2)
    else:
        return midpoint(p4, p1)


def process_image(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    objects = parse_dota_label_file(label_path)

    for obj in objects:
        pts = obj["points"]
        class_name = obj["class_name"]

        if class_name not in VEHICLE_CLASSES:
            continue

        center = polygon_center(pts)
        target = heading_target_from_p1(pts)

        cx, cy = int(round(center[0])), int(round(center[1]))
        tx, ty = int(round(target[0])), int(round(target[1]))

        cv2.arrowedLine(
            image,
            (cx, cy),
            (tx, ty),
            (0, 0, 255),
            2,
            tipLength=0.35
        )

    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description="Draw only vehicle arrows on DOTA images.")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--labels", required=True, help="Folder containing DOTA txt labels")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ])

    for image_file in image_files:
        image_path = os.path.join(args.images, image_file)
        base = os.path.splitext(image_file)[0]
        label_path = os.path.join(args.labels, base + ".txt")

        if not os.path.exists(label_path):
            print(f"Missing label for {image_file}, skipping.")
            continue

        output_path = os.path.join(args.output, base + "_arrows.jpg")
        process_image(image_path, label_path, output_path)
        print(f"Processed: {image_file}")


if __name__ == "__main__":
    main()
