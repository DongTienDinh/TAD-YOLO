# Kaggle T4 ablation notebooks

Run each notebook in a separate Kaggle session with **GPU enabled** and
**Internet on**. Add the YOLO-formatted VisDrone dataset as the only Kaggle
Input. The notebooks clone this repository, install the pinned runtime
dependencies without installing the project package (compatible with Kaggle's
Python 3.13 kernel), validate the dataset, run a model smoke test, and train
one frozen variant directly from the checked-out source tree.

The dataset YAML shipped inside the Kaggle Input is not used directly because
Kaggle may normalize the dataset slug (for example, `visdrone-det-yolo` to
`visdrone_det_yolo`). Each notebook generates
`/kaggle/working/visdrone_tad_yolo.yaml` from the actual mounted directory.

Run order:

1. `A0_YOLOv8n.ipynb`
2. `A1_P2.ipynb`
3. `A2_Shallow_Heavy.ipynb`
4. `A3_Terminal_P2.ipynb`
5. `A4_No_D5.ipynb`
6. `A5_TAD_YOLO.ipynb`

Each notebook defaults to seed 0. For a three-seed study, rerun each notebook
with seeds 42 and 123 and use a distinct run name. Outputs are written below
`/kaggle/working/runs/ablation` and archived to a ZIP at the end.

The GitHub `kaggle-ablation` branch must contain the ablation configs and
`scripts/train_ablation.py` before these notebooks are started.
