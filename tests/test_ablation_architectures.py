"""Contract tests for the six cumulative ablation graphs."""

from types import SimpleNamespace
import unittest

import torch

from tad_yolo.ablation import VARIANTS, build_variant
from tad_yolo.config import load_config
from tad_yolo.models.detector import TADYOLODetector


class AblationArchitectureTests(unittest.TestCase):
    def test_all_variants_match_parameter_and_stride_contracts(self) -> None:
        for key, variant in VARIANTS.items():
            with self.subTest(variant=key):
                model = build_variant(key)
                params = sum(parameter.numel() for parameter in model.model.parameters())
                strides = tuple(int(value) for value in model.model.stride)
                self.assertEqual(params, variant.expected_params)
                self.assertEqual(strides, variant.expected_strides)

    def test_a5_matches_fixed_tad_yolo_parameter_count(self) -> None:
        ablation = build_variant("a5_tad_yolo")
        fixed = TADYOLODetector.from_config(
            load_config("configs/tad_yolov8n_f4_paper300.yaml")
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in ablation.model.parameters()),
            sum(parameter.numel() for parameter in fixed.parameters()),
        )

    def test_a5_yaml_loss_and_backward_are_finite(self) -> None:
        model = build_variant("a5_tad_yolo").model.train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        batch = {
            "img": torch.rand(1, 3, 64, 64),
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
