# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmocr.registry import MODELS

from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class FCENet(SingleStageTextDetector):
    """The class for implementing FCENet text detector
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
        Detection

    [https://arxiv.org/abs/2104.10442]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter: int = 0
        self._acc_time: float = 0.0

    def __call__(self, *args, **kwargs):
        start = time.time()
        results = super().__call__(*args, **kwargs)
        self._acc_time += time.time() - start
        self._counter += 1
        if self._counter % 50:
            print(f"FCENet time: {self._counter / self._acc_time} fps each")
            self._counter = 0
        return results
