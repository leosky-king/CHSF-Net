

import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")


def initialize_dual_backbone(model, pretrained):
    
    source, target = pretrained.model.state_dict(), model.model.state_dict()
    copied = 0
    for source_index in range(11):
        prefix = f"model.{source_index}."
        for source_key, value in source.items():
            if not source_key.startswith(prefix):
                continue
            suffix = source_key[len(prefix):]
            for target_index in (source_index + 3, source_index + 14):
                target_key = f"model.{target_index}.{suffix}"
                if target_key in target and target[target_key].shape == value.shape:
                    target[target_key] = value.detach().clone()
                    copied += 1
    model.model.load_state_dict(target, strict=False)
    return copied


model = YOLO("improve_multimodal/yolo11/yolo11-CHSF-Net.yaml")
pretrained = YOLO("yolo11n.pt")
print(f"Initialized {initialize_dual_backbone(model, pretrained)} dual-backbone tensors from yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.8,
    weight_decay=0.001,
    cos_lr=True,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    amp=False,
    workers=8,
    close_mosaic=0,
    device="0",
)
