# TAD-YOLO cumulative ablation configurations

All six variants are trained from scratch with the same protocol in
`configs/tad_yolov8n_f4_paper300.yaml`. Each row adds exactly the named change.

| Key | Change | Detection strides | Parameter contract |
|---|---|---|---:|
| `a0_yolov8n` | frozen YOLOv8n baseline | 8, 16, 32 | 3,012,798 |
| `a1_p2` | A0 + P2 | 4, 8, 16, 32 | 2,927,880 |
| `a2_shallow_heavy` | A1 + 24/48/64/96/128 allocation | 4, 8, 16, 32 | 1,633,136 |
| `a3_terminal_p2` | A2 + terminal P2 | 4, 8, 16, 32 | 1,633,552 |
| `a4_no_d5` | A3 - D5 | 4, 8, 16 | 1,199,142 |
| `a5_tad_yolo` | A4 + two DySample operators | 4, 8, 16 | 1,204,326 |

The A5 parameter count is the executable current implementation's count. It
matches `TADYOLODetector` exactly; the preliminary paper table's historical
1.290M value does not match the source currently present in this repository.
