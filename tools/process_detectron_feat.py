import os
import numpy as np

detectron_feat_dir = '/private/home/nvivek/VQA/training_data/detectron_23/fc6/val'

rcnn_feat_dir = '/private/home/nvivek/VQA/training_data/rcnn_adaptive_vizwiz/vizwiz_ocr_att_sup_soft'



for feat_file in os.listdir(detectron_feat_dir):
    detectron_feat = np.load(os.path.join(detectron_feat_dir, feat_file))
    rcnn_feat = np.load(os.path.join(rcnn_feat_dir, feat_file))
    image_feat = detectron_feat
    image_text = rcnn_feat.item()['image_text']
    image_bbox_source = rcnn_feat.item()['image_bbox_source']
    image_text_vector = rcnn_feat.item()['image_text_vector']

    image_feat_and_boxes = {"image_feat": image_feat,
                            "image_text": image_text,
                            "image_bbox_source": image_bbox_source,
                            "image_text_vector": image_text_vector}

    np.save(os.path.join(detectron_feat_dir, feat_file), image_feat_and_boxes)
