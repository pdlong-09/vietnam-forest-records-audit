# Code and data: a design-based audit of Vietnam's 2020 forest status records

Companion release of the article "How Well Do National Forest Records Reflect Land Use on the Ground? A Design-Based
Audit of Vietnam's 2020 Forest Status Records and the EUDR Reference Maps".

## Quick check

The main estimates (Table 1 and the overall accuracies of Table 2) can be recomputed from the public sample alone:

    pip install numpy pandas
    python code/scripts/reproduce_from_release.py data

## Contents

- `data/`: the reference sample, also published on Kaggle
  (https://www.kaggle.com/datasets/linhnguyendinhmanh/vietnam-2020-forest-records-audit-reference-sample):
  `strata.csv` (12 phase-one strata with area and sample size), `phase1_points.csv` (20,000 phase-one points),
  `phase2_points_labels.csv` (1198 phase-two points with design weights, map calls and final reference labels) and
  `interpretations.csv` (every individual interpretation, interpreters coded A1-A4). Coordinates are rounded to
  0.01 degree; owner names and lot identifiers are not released. Column definitions: `data/README` on Kaggle and
  Supplementary Materials S2.
- `predictions/`: cross-fitted model predictions on the phase-one points of recorded forest, keyed by `pid`:
  `vlm_qwen35_9b_qlora.csv` (six-class land-use probabilities of the fine-tuned vision-language model, mean of the five
  fold adapters for points outside phase two, out of fold for phase-two points) and `cnn_resnet18_agriculture.csv`
  (probability of agricultural use, ResNet-18 on the same Sentinel-2 tiles). The AlphaEarth Foundations embeddings are
  not redistributed; `code/scripts/explore_b.py` fetches them from Google Earth Engine.
- `results/`: the result tables written by the analysis scripts (RQ1-RQ3, map accuracies, version changes, the
  AI-assisted estimates).
- `docs/interpretation_guide_vi.md`: the interpretation guide given to the interpreters (Vietnamese); the English
  protocol is in Supplementary Materials S2.
- `code/scripts/`: sampling (`draw_phase1.py`, `build_phase2.py`), label quality control and final labels
  (`qc_labels.py`, `draw_reinterpretation.py`, `qc_reinterp.py`, `build_final_labels.py`), map values and terrain from
  Earth Engine (`sample_maps_gee.py`), the two-phase estimators and RQ1-RQ3 (`analysis_b.py`), prediction-powered
  inference (`ppi_b.py`, `explore_b.py`), evidence cards and the VLM driver (`vlmb_build.py`, `vlmb_run.py`), figures
  and supplementary tables. `code/server/`: QLoRA training and inference of the VLM and the CNN on the GPU server.

Most scripts read the full 2020 forest status records and the raw interpretation sheets, which contain owner names and
lot identifiers and are therefore not included; the records themselves are public on Vietnam's Forestry 4.0 system
(https://tnr.vnforest.gov.vn). The GPU stages ran on two remote machines, whose host names and paths are replaced by
placeholders (`server1`, `server2`, `/home/user`); passwords are read from the environment and are not included.
Comments that mention "the analysis plan" or "the results notes" refer to internal planning documents, fixed before
each analysis was run, that are not part of the release. Analyses marked post hoc were added after the planned ones.

## Prompts of the vision-language model

The full prompt (evidence card text, the six options in cyclic order and the instruction) is built by `facts()` and
`prompt()` in `code/scripts/vlmb_run.py`; the evidence card images are rendered by `code/scripts/vlmb_build.py`.

## Development

The code was written with the help of an AI coding assistant (Claude, Anthropic), as stated in the article; the
authors designed the analyses and checked every result against the data.

## Licence

Code: MIT licence. Data, predictions and results: CC BY 4.0 (please cite the article and the sources listed in its
Data Availability statement).
