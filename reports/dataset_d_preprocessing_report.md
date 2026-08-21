# Dataset D Preprocessing Report (Stage 6A)

Generated: 2026-08-20 23:13:14 UTC

## 1. Input

- `Data/raw/dataset_d_eodhd/dataset_d_ohlcv.csv` (SHA-256 unchanged by this run: CONFIRMED)
- Raw Dataset D was neither re-downloaded, repaired, nor modified. The known KR adjusted-close provider break on 2022-10-21 is deliberately retained: this run measures its log return as 0.2113 and fails if it has disappeared.

## 2. Shared weekday calendar

- 3676 Monday-to-Friday dates, 2009-01-02 .. 2023-02-03; weekends excluded, public holidays retained as calendar dates.
- Dates added by reindexing (weekdays absent from the raw union calendar): 71
- All 13 M6 Friday anchors are present.

## 3. Forward filling

- Applied per asset per field, only after that asset's first genuine observation. No backward filling, interpolation or cross-asset substitution.
- Total cells forward-filled across all six fields: 77868
- Assets with genuine later inception (27): pre-inception weekdays remain missing.

| symbol | first genuine date | leading missing weekdays |
|---|---|---|
| ABBV | 2012-12-10 | 1026 |
| ALLE | 2013-11-18 | 1271 |
| CARR | 2020-03-19 | 2924 |
| CDW | 2013-06-27 | 1169 |
| CHTR | 2010-01-05 | 262 |
| DG | 2009-11-13 | 225 |
| FTV | 2016-06-13 | 1941 |
| HIGH.L | 2017-09-25 | 2276 |
| IEAA.L | 2017-09-25 | 2276 |
| IEFM.L | 2015-01-19 | 1576 |
| IEMG | 2012-10-22 | 991 |
| IEVL.L | 2015-01-19 | 1576 |
| INDA | 2012-02-03 | 805 |
| IUMO.L | 2016-11-03 | 2044 |
| IUVL.L | 2016-10-17 | 2031 |
| JPEA.L | 2017-04-13 | 2159 |
| MCHI | 2011-03-31 | 584 |
| META | 2012-05-18 | 880 |
| MVEU.L | 2012-12-11 | 1027 |
| OGN | 2021-05-14 | 3225 |
| PYPL | 2015-07-06 | 1696 |
| REET | 2014-07-10 | 1439 |
| SEGA.L | 2009-06-04 | 109 |
| SPMV.L | 2013-01-03 | 1044 |
| VRSK | 2009-10-07 | 198 |
| VXX | 2018-01-18 | 2359 |
| XLC | 2018-06-19 | 2467 |

- DRE: final genuine price 48.2000 on 2022-10-03, carried forward over 89 subsequent weekdays; every later log return is exactly zero (the established project treatment).

## 4. Returns

- Basis: ADJUSTED CLOSE. `r[t] = log(P[t] / P[t-1])`.
- Raw close is NOT used for returns because it contains mechanical split jumps that are not investment returns.
- Valid log returns: 327920; missing: 39680 (pre-inception plus each asset's first calendar row).
- No infinities. No clipping, winsorising, ECOD or normalisation.

## 5. Zero-volume observations (recorded, not modified)

- Raw zero-volume asset-days: 1946; after calendar alignment and forward filling: 2122.
- Handled locally in feature construction only (see `scripts/build_feature_baseline_dataset.py`); raw values are untouched.

| symbol | raw zero-volume days | zero-volume days after filling |
|---|---|---|
| CHTR | 18 | 18 |
| CNC | 1 | 1 |
| DRE | 1 | 90 |
| HIGH.L | 103 | 113 |
| IEAA.L | 65 | 70 |
| IEFM.L | 548 | 566 |
| IEVL.L | 270 | 280 |
| IUVL.L | 28 | 28 |
| JPEA.L | 49 | 53 |
| MCHI | 1 | 2 |
| SEGA.L | 782 | 814 |
| SPMV.L | 77 | 84 |
| VXX | 3 | 3 |
