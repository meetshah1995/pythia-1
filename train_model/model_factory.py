from global_variables.global_variables import *
from top_down_bottom_up.top_down_bottom_up_model import vqa_multi_modal_model
from top_down_bottom_up.image_attention import build_image_attention_module
from top_down_bottom_up.classifier import build_classifier
from top_down_bottom_up.question_embeding import build_question_encoding_module
from top_down_bottom_up.image_embedding import image_embedding
from top_down_bottom_up.multi_modal_combine import build_modal_combine_module
from top_down_bottom_up.intermediate_layer import inter_layer
from top_down_bottom_up.image_feature_encoding import build_image_feature_encoding
import torch.nn as nn
import json


def get_two_layer(img_dim):
    return inter_layer(img_dim, 2)


def prepare_model(num_vocab, num_choices, **model_config):
    print("Vocab " + str(num_vocab))
    print("Answers " + str(num_choices))
    image_feat_dim = model_config['image_feat_dim']
    with open('model_config', 'w') as f:
        config_str = json.dumps(model_config)
        f.write(config_str)

    ## generate the list of question embedding models
    ques_embeding_models_list = model_config['question_embedding']
    question_embeding_models = nn.ModuleList()
    final_question_embedding_dim = 0
    for ques_embeding_model in ques_embeding_models_list:
        ques_model_key = ques_embeding_model['method']
        ques_model_par = ques_embeding_model['par']
        tmp_model = build_question_encoding_module(ques_model_key, ques_model_par, num_vocab=num_vocab)

        question_embeding_models.append(tmp_model)
        final_question_embedding_dim += tmp_model.text_out_dim

    ## Image text feature encoding
    image_text_feat_embedding_models_list = None
    image_text_feature_encode_list = None
    image_text_feat_emb_dim = None
    final_image_text_feat_embedding_dim = None

    if model_config['use_image_text_feat']:
        image_text_feat_dim = model_config['image_text_feat_dim']
        image_text_feature_encode_list = nn.ModuleList()
        for image_text_feat_model_par in model_config['image_text_feat_encoding']:
            image_text_feat_model = build_image_feature_encoding(image_text_feat_model_par['method'],
                                                                 image_text_feat_model_par['par'],
                                                                 image_text_feat_emb_dim, num_vocab)
            image_text_feature_encode_list.append(image_text_feat_model)
            image_text_feat_dim = image_text_feat_model.out_dim

        image_text_feat_embedding_models_list = nn.ModuleList()
        num_image_text_feat = model_config['num_image_text_feat']
        final_image_text_feat_embedding_dim = 0
        for itf_image in range(num_image_text_feat):
            image_text_feat_embedding_models = nn.ModuleList()
            image_text_feat_att_model_list = model_config['image_text_feat_embedding_models']

            for image_text_feat_att_model in image_text_feat_att_model_list:
                image_text_feat_att_model_par = image_text_feat_att_model
                tmp_img_text_feat_att_model = build_image_attention_module(image_text_feat_att_model_par,
                                                                           image_dim=image_text_feat_dim,
                                                                           ques_dim=final_question_embedding_dim)
                tmp_img_text_model = image_embedding(tmp_img_text_feat_att_model)
                final_image_text_feat_embedding_dim += tmp_img_text_model.out_dim
                image_text_feat_embedding_models.append(tmp_img_text_model)
            image_text_feat_embedding_models_list.append(image_text_feat_embedding_models)

        final_image_text_feat_embedding_dim *= image_text_feat_dim

    image_feature_encode_list = nn.ModuleList()
    for image_feat_model_par in model_config['image_feature_encoding']:
        image_feat_model = build_image_feature_encoding(image_feat_model_par['method'],
                                                        image_feat_model_par['par'], image_feat_dim)
        image_feature_encode_list.append(image_feat_model)
        image_feat_dim = image_feat_model.out_dim

    ## generate the list of image attention models
    image_emdedding_models_list = nn.ModuleList()
    num_image_feat = model_config['num_image_feat']
    final_image_embedding_dim = 0
    for i_image in range(num_image_feat):
        image_emdedding_models = nn.ModuleList()
        image_att_model_list = model_config['image_embedding_models']

        for image_att_model in image_att_model_list:
            image_att_model_par = image_att_model
            tmp_img_att_model = build_image_attention_module(image_att_model_par,
                                                             image_dim=image_feat_dim,
                                                             ques_dim=final_question_embedding_dim)

            tmp_img_model = image_embedding(tmp_img_att_model)
            final_image_embedding_dim += tmp_img_model.out_dim
            image_emdedding_models.append(tmp_img_model)
        image_emdedding_models_list.append(image_emdedding_models)

    final_image_embedding_dim *= image_feat_dim

    inter_model = None
    # if num_image_feat > 1:
    #     inter_model=get_two_layer(final_image_embedding_dim)

    ##parse multi-modal combination after image-embedding and question-embedding
    multi_modal_combine = build_modal_combine_module(model_config['modal_combine']['method'],
                                                     model_config['modal_combine']['par'],
                                                     final_image_embedding_dim,
                                                     final_question_embedding_dim,
                                                     final_image_text_feat_embedding_dim)

    joint_embedding_dim = multi_modal_combine.out_dim
    ## generate the classifier
    classifier = build_classifier(model_config['classifier']['method'],
                                  model_config['classifier']['par'],
                                  in_dim=joint_embedding_dim, out_dim=num_choices)
    print("Building final model")
    print(image_emdedding_models_list)
    print(image_feature_encode_list)
    print(image_text_feat_embedding_models_list)
    print(image_text_feature_encode_list)
    my_model = vqa_multi_modal_model(image_emdedding_models_list,
                                     question_embeding_models,
                                     multi_modal_combine,
                                     classifier,
                                     image_feature_encode_list,
                                     inter_model,
                                     image_text_feat_embedding_models_list,
                                     image_text_feature_encode_list)
    print("Done")
    if use_cuda:
        my_model = my_model.cuda()

    if torch.cuda.device_count() > 1:
        my_model = nn.DataParallel(my_model)

    return my_model
