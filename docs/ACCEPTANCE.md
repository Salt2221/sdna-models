# Acceptance check

A sample PPM image was analysed locally with:

`python analyze.py --input sample.ppm --output results`

Verified output files:

- `results/annotated.png`
- `results/edges.png`
- `results/report.json`

The JSON includes image dimensions, four dominant color clusters, and detected visual regions when the image contains regions above the configured minimum area.
