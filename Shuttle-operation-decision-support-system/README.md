# Shuttle operation decision support system using surrogate model with graph-theoretic features

Code and data reproducing the results of Smid & Noda, AROB 2026.

## Run

Python 3.14.0:

```
pip install -r requirements.txt
python evaluate.py
```

The output matches `expected_output.txt`.

## Data

`data/train.jsonl` (7,200 rows) and `data/test.jsonl` (744 scenario groups × 12 candidate splits) contain one simulated (scenario, split) pair per row: fleet size `N`, shuttles `k`, requests `DEM`, simulated cancellation ratio `CR` and the graph-theoretic features. The simulator is not included, because scenario generation requires licensed Agoop mobility data.
