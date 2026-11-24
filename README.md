# RetiProteomics

Test code for a fundus-image model that predicts the prior (baseline) and
incident (follow-up) status of 32 diseases. The image encoder is the vision
encoder of [RetiZero](https://github.com/LooKing9218/RetiZero), fine-tuned with
two prediction heads (prior, incident) per disease.

Code only: no data or model weights are included.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python test.py --checkpoint best_model.pth \
    --test_csv test.csv --val_csv val.csv \
    --image_dir images/ --output_dir results/
```

- `--image_dir`: 224 × 224 fundus images named `<sno>_21015_0_0.png` (left eye)
  and `<sno>_21016_0_0.png` (right eye).
- `--test_csv`: one row per participant with columns `sno`, `prior_<disease>`
  and `incident_<disease>` (0/1). The disease names are stored in the checkpoint.
- `--val_csv` (optional): used to choose the decision threshold of each disease;
  otherwise 0.5 is used.

Results are written to `predictions.csv` (probabilities per participant,
averaged over both eyes) and `metrics.csv` (AUC, accuracy and F1 per disease).

## Data availability

The model was developed with UK Biobank data, which cannot be redistributed.
Access can be requested at <https://www.ukbiobank.ac.uk>.

## Acknowledgements

The code in `clip_modules/` is from [RetiZero](https://github.com/LooKing9218/RetiZero),
which builds on [RETFound](https://github.com/rmaphoh/RETFound_MAE) and
[timm](https://github.com/huggingface/pytorch-image-models).
