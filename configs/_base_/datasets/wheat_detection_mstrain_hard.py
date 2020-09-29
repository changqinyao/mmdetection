_base_ = "./wheat_detection_mstrain_light.py"

data_root = "/home/ubuntu/data/global-wheat-detection/"
data = dict(
    train=dict(
        ann_file=[
            data_root + "folds_v2/{fold}/coco_tile_train.json",
            data_root + "folds_v2/{fold}/coco_pseudo_train.json",
            data_root + "coco_spike.json",
        ],
        img_prefix=[
            dict(
                roots=[
                    data_root + "train/",
                    data_root + "colored_train/",
                    data_root + "stylized_train_v1/",
                    data_root + "stylized_train_v2/",
                    data_root + "stylized_train_v3/",
                    data_root + "stylized_train_v4/",
                ],
                probabilities=[0.5, 0.3, 0.2 / 4, 0.2 / 4, 0.2 / 4, 0.2 / 4],
            ),
            dict(
                roots=[
                    data_root + "crops_fold0/",
                    data_root + "colored_crops_fold0/",
                    data_root + "stylized_crops_fold0_v1/",
                    data_root + "stylized_crops_fold0_v2/",
                    data_root + "stylized_crops_fold0_v3/",
                    data_root + "stylized_crops_fold0_v4/",
                ],
                probabilities=[0.5, 0.3, 0.2 / 4, 0.2 / 4, 0.2 / 4, 0.2 / 4],
            ),
            dict(
                roots=[
                    data_root + "SPIKE_images/",
                    data_root + "stylized_SPIKE_images_v1/",
                    data_root + "stylized_SPIKE_images_v2/",
                    data_root + "stylized_SPIKE_images_v3/",
                    data_root + "stylized_SPIKE_images_v4/",
                ],
                probabilities=[0.7, 0.3 / 4, 0.3 / 4, 0.3 / 4, 0.3 / 4],
            ),
        ],
    )
)

