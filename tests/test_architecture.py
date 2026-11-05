"""Architecture-contract tests for the clean F4 implementation."""

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
import yaml

from tad_yolo.config import _validate_architecture_contract, load_config
from tad_yolo.models.detector import TADYOLODetector
from tad_yolo.nn.upsampling import DySample
from tad_yolo.offline import (
    find_ablation_checkpoint,
    validate_dataset_split,
    validate_dataset_yaml,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tad_yolov8n_f4.yaml"


def build_model() -> TADYOLODetector:
    return TADYOLODetector.from_config(load_config(CONFIG))


class ArchitectureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_fixed_f4_structure(self) -> None:
        model = build_model()
        self.assertEqual(model.model_name, "TAD-YOLO-F4")
        self.assertEqual(model.strides, (4.0, 8.0, 16.0))
        self.assertEqual(model.head_channels, (48, 64, 96))
        self.assertIsInstance(model.neck.t4_to_t3_upsample, DySample)
        self.assertIsInstance(model.neck.t3_to_t2_upsample, DySample)
        self.assertFalse(hasattr(model.neck, "t2_to_n3_downsample"))
        self.assertFalse(hasattr(model.neck, "n4_to_n5_downsample"))

    def test_feature_shapes(self) -> None:
        model = build_model().eval()
        for height, width in ((64, 64), (64, 96), (96, 128)):
            with self.subTest(height=height, width=width), torch.no_grad():
                features = model.forward_features(torch.randn(1, 3, height, width))
                expected = (
                    (1, 48, height // 4, width // 4),
                    (1, 64, height // 8, width // 8),
                    (1, 96, height // 16, width // 16),
                )
                self.assertEqual(tuple(tuple(feature.shape) for feature in features), expected)

    def test_zero_offset_dysample_matches_bilinear_resize(self) -> None:
        upsample = DySample(in_channels=8, scale=2, groups=4, offset_init_std=0.0).eval()
        x = torch.randn(1, 8, 7, 11)
        with torch.no_grad():
            actual = upsample(x)
            expected = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_loss_and_backward_are_finite(self) -> None:
        model = build_model().train()
        batch = {
            "img": torch.rand(1, 3, 64, 64),
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_config_rejects_non_f4_topology(self) -> None:
        config = load_config(CONFIG)
        config["model"]["has_d5"] = True
        with self.assertRaisesRegex(ValueError, "fixed F4"):
            _validate_architecture_contract(config)

    def test_dataset_yaml_falls_back_to_its_mounted_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "VisDrone2019-DET-train/images").mkdir(parents=True)
            (root / "VisDrone2019-DET-val/images").mkdir(parents=True)
            dataset_yaml = root / "visdrone_kaggle.yaml"
            dataset_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": "/kaggle/input/visdrone-det-yolo",
                        "train": "VisDrone2019-DET-train/images",
                        "val": "VisDrone2019-DET-val/images",
                        "names": {index: str(index) for index in range(10)},
                    }
                ),
                encoding="utf-8",
            )
            resolved = validate_dataset_yaml(dataset_yaml)
            self.assertEqual(Path(resolved["root"]), root.resolve())

    def test_labeled_test_split_and_checkpoint_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (
                "VisDrone2019-DET-train/images",
                "VisDrone2019-DET-val/images",
                "VisDrone2019-DET-test-dev/images",
                "VisDrone2019-DET-test-dev/labels",
            ):
                (root / directory).mkdir(parents=True)
            (root / "VisDrone2019-DET-test-dev/labels/000001.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
            dataset_yaml = root / "visdrone.yaml"
            dataset_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "VisDrone2019-DET-train/images",
                        "val": "VisDrone2019-DET-val/images",
                        "test": "VisDrone2019-DET-test-dev/images",
                        "names": {index: str(index) for index in range(10)},
                    }
                ),
                encoding="utf-8",
            )
            split = validate_dataset_split(dataset_yaml, "test", require_labels=True)
            self.assertEqual(Path(split["images"]), (root / "VisDrone2019-DET-test-dev/images").resolve())
            self.assertEqual(Path(split["labels"]), (root / "VisDrone2019-DET-test-dev/labels").resolve())

            checkpoint = root / "a4_no_d5_seed0/weights/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(
                find_ablation_checkpoint("a4_no_d5", input_root=root),
                checkpoint.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
