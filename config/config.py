import logging
from config.collections import AttrDict
from config.function_config_lib import ModelParPair

logger = logging.getLogger(__name__)

__C = AttrDict()
cfg = __C

# ---------------------------------------------------------------------------- #
# running model options: string, options: train, train+predict
# ---------------------------------------------------------------------------- #
__C.run = "train+predict"
__C.exp_name = "baseline"

# ---------------------------------------------------------------------------- #
# data options:
# ---------------------------------------------------------------------------- #
__C.data = AttrDict()

__C.data.dataset = "vizwiz"
__C.data.num_workers = 5
__C.data.batch_size = 512
__C.data.image_depth_first = False
__C.data.question_max_len = 14
__C.data.image_fast_reader = False
__C.data.image_max_loc = 137

__C.data.data_root_dir = "/private/home/nvivek/VQA/training_data/"

__C.data.vocab_question_file = "vocabulary_glove.txt"
__C.data.vocab_answer_file = "answers_vizwiz.txt"

__C.data.image_feat_train = ["rcnn_adaptive_vizwiz/vizwiz"]
__C.data.imdb_file_train = ["imdb/imdb_vizwiz_train.npy"]

__C.data.imdb_file_val = ["imdb/imdb_vizwiz_val.npy"]
__C.data.image_feat_val = ["rcnn_adaptive_vizwiz/vizwiz"]

__C.data.imdb_file_test = ["imdb/imdb_vizwiz_test.npy"]
__C.data.image_feat_test = ["rcnn_adaptive_vizwiz/vizwiz"]


# ---------------------------------------------------------------------------- #
# training_parameters options:
# ---------------------------------------------------------------------------- #
__C.training_parameters = AttrDict()
__C.training_parameters.report_interval = 100
__C.training_parameters.snapshot_interval = 1000
__C.training_parameters.clip_norm_mode = 'all'
__C.training_parameters.max_grad_l2_norm = 0.25
__C.training_parameters.wu_factor = 0.2
__C.training_parameters.wu_iters = 1000
__C.training_parameters.max_iter = 12000
__C.training_parameters.lr_steps = [3000, 6000, 8000]
__C.training_parameters.lr_ratio = 0.1


# ---------------------------------------------------------------------------- #
# loss options:
# ---------------------------------------------------------------------------- #

__C.loss = 'logitBCE'


# ---------------------------------------------------------------------------- #
# optimizer options:
# ---------------------------------------------------------------------------- #

__C.optimizer = ModelParPair('Adamax')


# ---------------------------------------------------------------------------- #
# model options: Note default is our
# ---------------------------------------------------------------------------- #

__C.model = AttrDict()
__C.model.use_image_text_feat = False
__C.model.image_feat_dim = 2048
__C.model.image_text_feat_dim = 300
__C.model.num_image_text_feat = 1
__C.model.question_embedding = [ModelParPair("att_que_embed")]
__C.model.image_text_feat_encoding = [ModelParPair("image_text_feat_encoding")]
__C.model.image_feature_encoding=[ModelParPair('default_image')]
__C.model.image_embedding_models = []
__C.model.image_text_feat_embedding_models = []
__C.model.modal_combine = ModelParPair('non_linear_elmt_multiply')
__C.model.classifier = ModelParPair('logit_classifier')

top_down_bottom_up = AttrDict()
top_down_bottom_up.modal_combine = ModelParPair('non_linear_elmt_multiply')
top_down_bottom_up.transform = ModelParPair('linear_transform')
top_down_bottom_up.normalization = 'softmax'

__C.model.image_embedding_models.append(top_down_bottom_up)
__C.model.image_text_feat_embedding_models.append(top_down_bottom_up)
