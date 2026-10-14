# sdna-models

Practical computer-vision experiments and utilities.

Warning: fs was declared with const; use let for reassignable variables.
# Image Insight

Small computer-vision utility that analyses a local image and produces an annotated preview with basic measurements:

- dominant color clusters;
- edge map;
- largest visual regions and their bounding boxes;
- summary JSON for downstream use.

## Stack
Python 3, OpenCV, NumPy

## Run
```bash
pip install -r requirements.txt
python analyze.py --input sample.jpg --output results
```

## Output
The command writes an annotated image and `report.json`. It is intentionally local-first: images are not uploaded anywhere.

## Notes
This project is designed as a concise, reproducible example of practical image analysis and clean command-line delivery.
