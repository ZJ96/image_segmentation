import os
import torch
from .models.DeepLab_v3_plus.decoder import DeepLab
import ttach as tta

def init_model():
    path  = os.path.join(os.path.dirname(__file__), 'net.pth')
    model = DeepLab(output_stride=16,class_num=17,pretrained=False,bn_momentum=0.1,freeze_bn=False)
    model.load_state_dict(torch.load(path))

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
        ]
    )

    model = tta.SegmentationTTAWrapper(model, transforms)
    model = model.cuda()

    return model
