"""Ultralytics trainer adapter for the fixed TAD-YOLO graph."""

from __future__ import annotations

from copy import deepcopy

from ultralytics.models.yolo.detect import DetectionTrainer

from tad_yolo.models.detector import TADYOLODetector


class TADYOLOTrainer(DetectionTrainer):
    """Construct TAD-YOLO and strictly restore only matching checkpoints."""

    def __init__(self, experiment_config: dict, **kwargs) -> None:
        self.experiment_config = deepcopy(experiment_config)
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        expected_classes = int(self.experiment_config["model"]["num_classes"])
        if self.data["nc"] != expected_classes:
            raise ValueError(
                f"Dataset has {self.data['nc']} classes; config requires {expected_classes}."
            )
        model = TADYOLODetector.from_config(self.experiment_config)
        model.yaml = {**deepcopy(self.experiment_config), "nc": expected_classes, "channels": 3}
        model.names = self.data["names"]
        if weights is not None:
            if not isinstance(weights, TADYOLODetector):
                raise ValueError("Initialization/resume requires a TADYOLODetector checkpoint.")
            model.load_state_dict(weights.float().state_dict(), strict=True)
        return model
