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

            pts = np.array(
                coords, dtype=np.float32
            ).reshape(4, 2)

            objects.append({
                "points": pts,
                "class_name": parts[8],
                "difficult": parts[9]
            })

    return objects


def hddet_heading(pts):
    """
    HDDet heading calculation.

    Reference:
    G-virus/YOLOv6-HDDet
    yolov6/utils/8corners2yolo.py

    P1-P2 midpoint represents the annotated heading side.

    Angle:
      0 degrees   = North
      90 degrees  = East
      180 degrees = South
      270 degrees = West
    """

    x = pts[:, 0]
    y = pts[:, 1]

    # HDDet center: coordinate extrema midpoint
    x_center = (np.max(x) + np.min(x)) / 2.0
    y_center = (np.max(y) + np.min(y)) / 2.0

    center = np.array(
        [x_center, y_center],
        dtype=np.float32
    )

    # HDDet heading side: midpoint of P1-P2
    front_x = (x[0] + x[1]) / 2.0
    front_y = (y[0] + y[1]) / 2.0

    front = np.array(
        [front_x, front_y],
        dtype=np.float32
    )

    # Heading vector
    vx = front_x - x_center
    vy = front_y - y_center

    # Reference north vector
    v0x = 0.0
    v0y = -1.0

    cross = vx * v0y - vy * v0x
    dot = v0x * vx + v0y * vy

    # Match HDDet clockwise angle convention
    angle = (
        360.0 -
        ((math.degrees(math.atan2(cross, dot)) + 360.0) % 360.0)
    ) % 360.0

    return center, front, angle


def draw_points(image, pts):
    for i, point in enumerate(pts):
        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            image,
            (x, y),
            4,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            image,
            "P{}".format(i + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )


def process_image(
    image_path,
    label_path,
    output_path,
    show_points=False
):
    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read:", image_path)
        return []

    objects = parse_dota_label_file(label_path)

    rows = []

    for object_index, obj in enumerate(objects):

        class_name = obj["class_name"]

        if class_name not in VEHICLE_CLASSES:
            continue

        pts = obj["points"]

        # Draw oriented bounding box
        polygon = np.round(pts).astype(np.int32)

        cv2.polylines(
            image,
            [polygon],
            True,
            (0, 255, 0),
            2
        )

        # HDDet heading calculation
        center, front, angle = hddet_heading(pts)

        cx = int(round(center[0]))
        cy = int(round(center[1]))

        fx = int(round(front[0]))
        fy = int(round(front[1]))

        # Draw red direction arrow
        cv2.arrowedLine(
            image,
            (cx, cy),
            (fx, fy),
            (0, 0, 255),
            2,
            tipLength=0.35
        )

        # Draw center
        cv2.circle(
            image,
            (cx, cy),
            3,
            (255, 255, 255),
            -1
        )

        # Label with heading angle
        label = "{} {:.1f} deg".format(
            class_name, angle
        )

        cv2.putText(
            image,
            label,
            (cx + 5, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        if show_points:
            draw_points(image, pts)

        rows.append({
            "image": os.path.basename(image_path),
            "object_index": object_index,
            "class_name": class_name,
            "center_x": round(float(center[0]), 2),
            "center_y": round(float(center[1]), 2),
            "front_x": round(float(front[0]), 2),
            "front_y": round(float(front[1]), 2),
            "heading_angle_deg": round(float(angle), 2),
            "method": "HDDet_P1_P2_midpoint"
        })

    cv2.imwrite(output_path, image)

    return rows


def main():

    parser = argparse.ArgumentParser(
        description="Draw DOTA vehicle arrows using HDDet logic"
    )

    parser.add_argument(
        "--images",
        required=True
    )

    parser.add_argument(
        "--labels",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--show-points",
        action="store_true"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_files = sorted([
        filename
        for filename in os.listdir(args.images)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        )
    ])

    all_rows = []

    for image_file in image_files:

        base = os.path.splitext(image_file)[0]

        image_path = os.path.join(
            args.images, image_file
        )

        label_path = os.path.join(
            args.labels, base + ".txt"
        )

        if not os.path.isfile(label_path):
            print("Missing annotation:", image_file)
            continue

        output_path = os.path.join(
            args.output, base + "_hddet.jpg"
        )

        rows = process_image(
            image_path,
            label_path,
            output_path,
            args.show_points
        )

        all_rows.extend(rows)

        print(
            "Processed: {} | Vehicles: {}".format(
                image_file, len(rows)
            )
        )

    csv_path = os.path.join(
        args.output, "vehicle_headings.csv"
    )

    fieldnames = [
        "image",
        "object_index",
        "class_name",
        "center_x",
        "center_y",
        "front_x",
        "front_y",
        "heading_angle_deg",
        "method"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\nHDDet processing complete.")
    print("Images:", args.output)
    print("CSV:", csv_path)
    print("Total vehicles:", len(all_rows))


if __name__ == "__main__":
    main()
