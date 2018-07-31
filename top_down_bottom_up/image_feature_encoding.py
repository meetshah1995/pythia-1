import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from config.config import cfg


def build_image_feature_encoding(method, par, in_dim, num_vocab=None):
    if method == "default_image":
        return DefaultImageFeature(in_dim)
    elif method == "image_text_feat_encoding":
        return ImageTextFeatureEmbedding(num_vocab, **par)
    elif method == "finetune_faster_rcnn_fpn_fc7":
        return FinetuneFasterRcnnFpnFc7(in_dim, **par)
    else:
        raise NotImplementedError("unknown image feature encoding %s" % method)


class DefaultImageFeature(nn.Module):
    def __init__(self, in_dim):
        super(DefaultImageFeature, self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, image):
        return image

class ImageTextFeatureEmbedding(nn.Module):
    def __init__(self, num_vocab, **kwargs):
        super(ImageTextFeatureEmbedding, self).__init__()
        self.out_dim = kwargs['embedding_dim']
        self.num_vocab = num_vocab
        self.embedding = nn.Embedding(num_vocab, self.out_dim)
        # self.embedding = None
        # if 'embedding_init' in kwargs and kwargs['embedding_init'] is not None:
        #     self.embedding.weight.data.copy_(torch.from_numpy(kwargs['embedding_init']))

    def forward(self, input_text):
        input_shape = input_text.size()
        bs = input_text.size(0)
        feat_len = input_text.size(1)
        word_len = input_text.size(2)
        input_text = input_text.view(-1, word_len)
        embed_txt = self.embedding(input_text)
        embed_txt = embed_txt.view(*input_shape, -1)
        embed_txt = embed_txt.mean(2)
        return embed_txt

class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(cfg.data.data_root_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(cfg.data.data_root_dir, bias_file)
        with open(weights_file, 'rb') as w:
            weights = pickle.load(w)
        with open(bias_file, 'rb') as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = F.relu(i2)
        return i3
