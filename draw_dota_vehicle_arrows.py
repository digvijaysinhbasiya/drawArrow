import os
import csv
import math
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
            difficult = parts[9]

            objects.append({
                "points": pts,
                "class_name": class_name,
                "difficult": difficult
            })
    return objects


def polygon_center(pts):
    return np.mean(pts, axis=0)


def midpoint(a, b):
    return (a + b) / 2.0


def heading_target_from_p1(pts):
    """
    DOTA-inspired heuristic:
    Use the midpoint of the shorter edge adjacent to P1.
    Adjacent edges are P1-P2 and P4-P1.
    """
    p1, p2, p3, p4 = pts

    d12 = np.linalg.norm(p2 - p1)
    d41 = np.linalg.norm(p1 - p4)

    if d12 <= d41:
        return midpoint(p1, p2), "P1-P2"
    else:
        return midpoint(p4, p1), "P4-P1"


def heading_angle_deg(center, target):
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    angle = math.degrees(math.atan2(-dy, dx))
    return angle


def draw_points(img, pts):
    for i, p in enumerate(pts):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            img, f"P{i+1}", (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )


def process_image(image_path, label_path, output_path, show_points=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return []

    objects = parse_dota_label_file(label_path)
    rows = []

    for obj_idx, obj in enumerate(objects):
        pts = obj["points"]
        class_name = obj["class_name"]

        # Draw all objects lightly
        poly = np.round(pts).astype(np.int32)
        color = (0, 200, 0) if class_name in VEHICLE_CLASSES else (180, 180, 180)
        cv2.polylines(image, [poly], True, color, 2)

        if class_name not in VEHICLE_CLASSES:
            continue

        center = polygon_center(pts)
        target, front_edge = heading_target_from_p1(pts)

        cx, cy = int(round(center[0])), int(round(center[1]))
        tx, ty = int(round(target[0])), int(round(target[1]))

        # Draw center
        cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1)

        # Draw arrow
        cv2.arrowedLine(
            image,
            (cx, cy),
            (tx, ty),
            (0, 0, 255),
            2,
            tipLength=0.3
        )

        # Draw class label
        label_pos = (cx + 5, cy + 15)
        cv2.putText(
            image, class_name, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )

        if show_points:
            draw_points(image, pts)

        angle_deg = heading_angle_deg(center, target)

        rows.append({
            "image": os.path.basename(image_path),
            "object_index": obj_idx,
            "class_name": class_name,
            "center_x": round(float(center[0]), 2),
            "center_y": round(float(center[1]), 2),
            "arrow_x": round(float(target[0]), 2),
            "arrow_y": round(float(target[1]), 2),
            "heading_angle_deg": round(float(angle_deg), 2),
            "front_edge_used": front_edge
        })

    cv2.imwrite(output_path, image)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Draw DOTA vehicle direction arrows.")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--labels", required=True, help="Folder containing DOTA txt labels")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--show-points", action="store_true", help="Draw P1/P2/P3/P4 labels")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ])

    csv_rows = []

    for image_file in image_files:
        image_path = os.path.join(args.images, image_file)
        base = os.path.splitext(image_file)[0]
        label_path = os.path.join(args.labels, base + ".txt")

        if not os.path.exists(label_path):
            print(f"Missing label for {image_file}, skipping.")
            continue

        output_path = os.path.join(args.output, base + "_arrows.jpg")
        rows = process_image(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            show_points=args.show_points
        )
        csv_rows.extend(rows)
        print(f"Processed: {image_file}")

    csv_path = os.path.join(args.output, "vehicle_headings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "object_index", "class_name",
                "center_x", "center_y",
                "arrow_x", "arrow_y",
                "heading_angle_deg", "front_edge_used"
            ]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone. Output images saved in: {args.output}")
    print(f"CSV saved at: {csv_path}")


if __name__ == "__main__":
    main()
