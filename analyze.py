#!/usr/bin/env python3
"""Analyze a local image and export a compact computer-vision report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def dominant_colors(image: np.ndarray, k: int = 4) -> list[list[int]]:
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    ordered = np.argsort(counts)[::-1]
    return [centers[i].round().astype(int).tolist()[::-1] for i in ordered]


def detect_regions(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = max(250, int(h * w * 0.005))
    regions: list[dict[str, int]] = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = rw * rh
        if area >= min_area:
            regions.append({'x': int(x), 'y': int(y), 'width': int(rw), 'height': int(rh), 'area': int(area)})
    regions.sort(key=lambda item: item['area'], reverse=True)
    return edges, regions[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description='Create an image-analysis preview and JSON report.')
    parser.add_argument('--input', required=True, type=Path, help='Path to a local image')
    parser.add_argument('--output', default=Path('results'), type=Path, help='Directory for output files')
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise SystemExit(f'Cannot read image: {args.input}')
    args.output.mkdir(parents=True, exist_ok=True)

    edges, regions = detect_regions(image)
    annotated = image.copy()
    for index, region in enumerate(regions, start=1):
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (40, 210, 80), 2)
        cv2.putText(annotated, str(index), (x + 5, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 210, 80), 2)

    report = {
        'source': str(args.input),
        'image_size': {'width': int(image.shape[1]), 'height': int(image.shape[0])},
        'dominant_colors_rgb': dominant_colors(image),
        'regions': regions,
    }
    cv2.imwrite(str(args.output / 'annotated.png'), annotated)
    cv2.imwrite(str(args.output / 'edges.png'), edges)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Created {args.output / "annotated.png"}')


if __name__ == '__main__':
    main()
