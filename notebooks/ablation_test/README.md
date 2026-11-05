# Kaggle test evaluation notebooks

These six notebooks evaluate the `best.pt` selected by validation for each
ablation variant. They do not train or modify the checkpoint.

For each notebook, enable one T4 with Internet On and attach two Kaggle Inputs:

1. the labeled YOLO-format VisDrone dataset;
2. the saved output from the matching training notebook, including `best.pt`.

The notebook discovers one matching `best.pt` under `/kaggle/input`, verifies
its parameter/stride contract, requires labels for the `test` split, and runs
one evaluation with `imgsz=640`, `batch=8`, FP16, `conf=0.001`, `iou=0.7`, and
`max_det=300`. If several matching checkpoints are attached, set `CHECKPOINT`
in the first code cell to the intended absolute path.

Run the notebook corresponding to each trained model:

1. `A0_YOLOv8n_Test.ipynb`
2. `A1_P2_Test.ipynb`
3. `A2_Shallow_Heavy_Test.ipynb`
4. `A3_Terminal_P2_Test.ipynb`
5. `A4_No_D5_Test.ipynb`
6. `A5_TAD_YOLO_Test.ipynb`

Outputs are written below `/kaggle/working/runs/ablation_test`. Each run stores
Ultralytics metrics/plots plus `test_manifest.json`, which records the
checkpoint SHA-256, resolved test paths, evaluation settings, metrics, and
speed. A small ZIP of the evaluation directory is also created.
