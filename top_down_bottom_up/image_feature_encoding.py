# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from config.config import cfg
from top_down_bottom_up.utils import get_iou


def build_image_feature_encoding(method, par, in_dim):
    if method == "default_image":
        return DefaultImageFeature(in_dim)
    elif method == "finetune_faster_rcnn_fpn_fc7":
        return FinetuneFasterRcnnFpnFc7(in_dim, **par)
    elif method == "gcn_finetune_faster_rcnn_fpn_fc7":
        return GcnFinetuneFasterRcnnFpnFc7(in_dim, **par)
    else:
        raise NotImplementedError("unknown image feature encoding %s" % method)


class DefaultImageFeature(nn.Module):
    def __init__(self, in_dim):
        super(DefaultImageFeature, self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, image):
        return image


class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()

        if weights_file is not None and bias_file is not None:
            if not os.path.isabs(weights_file):
                weights_file = os.path.join(cfg.data.data_root_dir, weights_file)
            if not os.path.isabs(bias_file):
                bias_file = os.path.join(cfg.data.data_root_dir, bias_file)
            with open(weights_file, 'rb') as w:
                weights = pickle.load(w)
            with open(bias_file, 'rb') as b:
                bias = pickle.load(b)
            out_dim = bias.shape[0]
        else:
            out_dim = 2048

        self.lc = nn.Linear(in_dim, out_dim)

        if weights_file is not None and bias_file is not None:
            self.lc.weight.data.copy_(torch.from_numpy(weights))
            self.lc.bias.data.copy_(torch.from_numpy(bias))

        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = F.relu(i2)
        return i3



class GcnFinetuneFasterRcnnFpnFc7(FinetuneFasterRcnnFpnFc7):
    def __init__(self, 
                 in_dim, 
                 weights_file, 
                 bias_file, 
                 n_feats, 
                 relations=None,
                 branch_from=None):
        super(GcnFinetuneFasterRcnnFpnFc7, self).__init__(in_dim,
                                                          weights_file,
                                                          bias_file)

        self.require_bbox = True

        # relations for the graph e.g. ['LO', 'RO', 'TO', 'BO', 'IoU']
        self.relations = relations

        # See if need to use fc6 or fc7
        self.branch_from = branch_from
        
        # set output dimension
        self.out_dim = 2048

        # learnable weights for each relation L * F_out * F_out
        weight = torch.ones(len(self.relations), self.out_dim, 2048)
        nn.init.kaiming_uniform_(weight, a=2.233606)
        self.gcn_weights = nn.Parameter(weight, requires_grad=True)


    def _get_relation_matrices(self, bboxes, img_h, img_w):
        # bboxes = N * 4
        bs = len(bboxes)
        n_boxes = len(bboxes[1])
        relation_matrices = []

        mean_x = (bboxes[:,:,0] + bboxes[:,:,2]) / 2.0 / img_w.unsqueeze(-1).expand(bs, n_boxes).float()
        mean_y = (bboxes[:,:,1] + bboxes[:,:,3]) / 2.0 / img_h.unsqueeze(-1).expand(bs, n_boxes).float()

        mean_x = mean_x.unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1)

        diff_x = mean_x.expand(bs, n_boxes, n_boxes) - \
                 mean_x.expand(bs, n_boxes, n_boxes).permute([0,2,1])
        diff_y = mean_y.expand(bs, n_boxes, n_boxes) - \
                 mean_y.expand(bs, n_boxes, n_boxes).permute([0,2,1])

        for r in self.relations:
            if r == 'LO':
                g = F.sigmoid(diff_x).unsqueeze(1)
                relation_matrices.append(g)
            elif r == 'RO':
                g = F.sigmoid(-diff_x).unsqueeze(1)
                relation_matrices.append(g)
            elif r == 'TO':
                g = F.sigmoid(diff_y).unsqueeze(1)
                relation_matrices.append(g)
            elif r == 'BO':
                g = F.sigmoid(-diff_y).unsqueeze(1)
                relation_matrices.append(g)
            elif r == 'I':
                g = torch.eye(n_boxes).expand(bs, n_boxes, n_boxes)
                g = g.unsqueeze(1).to(mean_x.device)
                relation_matrices.append(g)
            elif r == 'IoU':
                iou_mats = []
                for _boxes in bboxes:
                    iou_mats.append(get_iou(_boxes, _boxes))
                g = torch.stack(iou_mats, 0)
                relation_matrices.append(g.unsqueeze(1))
            else:
                raise NotImplementedError

        # concat along second dim
        G = torch.cat(relation_matrices, 1)

        # normalize along the rows
        G = F.normalize(G, p=1, dim=3)

        return G
        

    def forward(self, image_feat, boxes, img_h, img_w):
        bs = len(image_feat)
        
        if self.branch_from == 'fc7':
            # Pass features through fc7 layer for finetuning
            gcn_input = super(GcnFinetuneFasterRcnnFpnFc7, self).forward(image_feat)
        else:
            gcn_input = image_feat
        
        # Get graph matric for each relation BS * L * N * N
        G = self._get_relation_matrices(boxes, img_h, img_w)
        
        # Compute modified features by multiplying matrices
        out = torch.stack([F.relu(F.linear(torch.bmm(G[:,i], gcn_input), 
                                    self.gcn_weights[i])) \
                           for i in range(len(self.relations))], 0)

        out = torch.mean(out, 0)
        return out
