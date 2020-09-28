from multiprocessing import Pool

import numpy as np
from mmcv.utils import print_log

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.evaluation.mean_ap import get_cls_results


def calc_tpfpfn(det_bboxes, gt_bboxes, iou_thr=0.5):
    """Check if detected bboxes are true positive or false positive and if gt bboxes are false negative.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        float: (tp, fp, fn).
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    tp = 0
    fp = 0

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp = num_dets
        return tp, fp, 0

    ious: np.ndarray = bbox_overlaps(det_bboxes, gt_bboxes)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        uncovered_ious = ious[i, gt_covered == 0]
        if len(uncovered_ious):
            iou_argmax = uncovered_ious.argmax()
            iou_max = uncovered_ious[iou_argmax]
            if iou_max >= iou_thr:
                gt_covered[[x[iou_argmax] for x in np.where(gt_covered == 0)]] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = (gt_covered == 0).sum()
    return tp, fp, fn


def kaggle_map(
    det_results, annotations, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), logger=None, n_jobs=4, by_sample=False
):
    """Evaluate kaggle mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        iou_thrs (list): IoU thresholds to be considered as matched.
            Default: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75).
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        n_jobs (int): Processes used for computing TP, FP and FN.
            Default: 4.
        by_sample (bool): Return AP by sample.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num

    pool = Pool(n_jobs)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _ = get_cls_results(det_results, annotations, i)
        # compute tp and fp for each image with multiple processes
        aps_by_thrs = []
        aps_by_sample = np.zeros(num_imgs)
        for iou_thr in iou_thrs:
            tpfpfn = pool.starmap(calc_tpfpfn, zip(cls_dets, cls_gts, [iou_thr for _ in range(num_imgs)]))
            iou_thr_aps = np.array([tp / (tp + fp + fn) for tp, fp, fn in tpfpfn])
            if by_sample:
                aps_by_sample += iou_thr_aps
            aps_by_thrs.append(np.mean(iou_thr_aps))
        eval_results.append(
            {
                "num_gts": len(cls_gts),
                "num_dets": len(cls_dets),
                "ap": np.mean(aps_by_thrs),
                "ap_by_sample": None if not by_sample else aps_by_sample / len(iou_thrs),
            }
        )
    pool.close()

    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_log(f"\nKaggle mAP: {mean_ap}", logger=logger)
    return mean_ap, eval_results

import torch

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def kaggle_map_yolo(
    det_results, annotations, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), logger=None, n_jobs=4, by_sample=False
):
    iouv=torch.tensor(iou_thrs)
    niou=len(iouv)
    seen=0
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for si, pred in enumerate(det_results):
        pred=torch.from_numpy(pred[0]) if pred else None
        tcls = torch.from_numpy(annotations[si]['labels'])
        tbox= torch.from_numpy(annotations[si]['bboxes'])
        nl = len(tcls)
        seen += 1

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Append to text file
        # with open('test.txt', 'a') as file:
        #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        if nl:
            detected = []  # target indices

            # Per target class
            for cls in np.unique(tcls):
                ti = (cls == tcls).nonzero().view(-1)  # prediction indices
                pi = torch.tensor([i for i in range(len(pred))])  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pred[pi,:4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    for j in (ious > iouv[0]).nonzero():
                        d = ti[i[j]]  # detected target
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        pcls=torch.zeros(pred.shape[0])
        stats.append((correct, pred[:, 4], pcls, tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
    else:
        nt = torch.zeros(1)
    return mp, mr, map, mf1