import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .evaluation import kaggle_map,kaggle_map_yolo
import pandas as pd
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class WheatDataset(CocoDataset):

    CLASSES = ('wheat')

    def evaluate(self, results, logger=None, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), **kwargs):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        mp, mr, map, mf1 = kaggle_map_yolo(results, annotations, iou_thrs=iou_thrs, logger=logger)
        return dict(mp=mp,mR=mr,mAP=map,mAP50=mf1)

    # def evaluate(self, results, logger=None, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), **kwargs):
    #     annotations = [self.get_ann_info(i) for i in range(len(self))]
    #     mean_ap, _ = kaggle_map(results, annotations, iou_thrs=iou_thrs, logger=logger)
    #     return dict(mAP=mean_ap)

    def format_results(self, results, output_path=None, **kwargs):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )
        prediction_results = []
        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]

            prediction_strs = []
            for bbox in wheat_bboxes:
                x, y, w, h = self.xyxy2xywh(bbox)
                prediction_strs.append(f"{bbox[4]:.4f} {x} {y} {w} {h}")
            filename = self.data_infos[idx]["filename"]
            image_id = osp.splitext(osp.basename(filename))[0]
            prediction_results.append({"image_id": image_id, "PredictionString": " ".join(prediction_strs)})
        predictions = pd.DataFrame(prediction_results)
        if output_path is not None:
            predictions.to_csv(output_path, index=False)
        return predictions