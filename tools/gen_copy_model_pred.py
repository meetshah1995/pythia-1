import argparse
import os
import random
import torch
import yaml
from torch.utils.data import DataLoader
from train_model.dataset_utils import prepare_test_data_set, prepare_eval_data_set, prepare_train_data_set
import numpy as np
from os import listdir
from dataset_utils import text_processing
from train_model.helper import run_model, build_model
import matplotlib.pyplot as plt
import cv2
import json
from random import shuffle

im_dir = '/private/home/nvivek/vizwiz/Images/'
config_file = '/checkpoint/nvivek/test_7k_answers_copy_only_ocr_partial_score/results/baseline_vizwiz_predict/54278/config.yaml'
model_file = '/checkpoint/nvivek/test_7k_answers_copy_only_ocr_partial_score/results/baseline_vizwiz_predict/54278/best_model.pth'
im_feat_dir = '/private/home/nvivek/VQA/training_data/rcnn_adaptive_vizwiz/vizwiz_att_ans'
test_annot_file = '/private/home/nvivek/vizwiz/Annotations/v2_mscoco_vizwiz_test_annotations.json'
train_annot_file = '/private/home/nvivek/VQA/training_data/v2_mscoco_vizwiz_train_annotations.json'
val_annot_file = '/private/home/nvivek/VQA/training_data/v2_mscoco_vizwiz_val_annotations.json'
max_loc = 137
att_sup_file = 'attention_sup.json'

with open(att_sup_file, 'r') as f:
    att_sups = json.load(f)

def visualize_att(im_file, im_feature, att, k, show=False):
    im = cv2.imread(im_file)
    im = im[:, :, (2, 1, 0)]
    if show:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im)

    im_data = np.load(im_feature)
    text = im_data.item().get('image_text').tolist()
    bbox = im_data.item().get('image_bboxes').tolist()
    bbox_source = im_data.item().get('image_bbox_source').tolist()

    top_k_idx = list(np.argsort(att)[-k:])
    best_text = ''

    for i,(source,bbox,text) in enumerate(zip(bbox_source, bbox, text)):

        edgecolor = 'red'
        if source[0] == 1:
            edgecolor = 'green'
        fill = False
        alpha = 0.5
        if i not in top_k_idx:
            continue
        if i == top_k_idx[-1]:
            best_text = text
        if not show:
            continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=fill, alpha=alpha,
                          edgecolor=edgecolor, linewidth=3.5)
            )
        box_text = '{:s} {:.3f}'.format(text, att[i])

        ax.text(bbox[0] + 5, bbox[1] + 5,
                box_text,
                bbox=dict(alpha=0.5),
                fontsize=14, color='white')

    # if show:
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.draw()
    #     plt_file_path = im_file.split('/')[-1].replace(".jpg", "_pltatt.jpg")
    #     plt.savefig(plt_file_path)
        # clear_output()
        # display(Image(filename=plt_file_path))
    return best_text

def get_image(source='test'):
    im_files = [f for f in os.listdir(im_dir) if source in f]
    im_file = random.choice(im_files)
    im_path = os.path.join(im_dir, im_file)
    print(im_path)
    # im_feature_path = os.path.join(im_feat_dir, 'COCO_' + im_file.split('/')[-1].replace(".jpg", ".npy").lower())
    # visualize(im_path, im_feature_path)
    # display(Image(filename=im_path))
    return im_file

def get_imdb(im_file, question_str):
    imdb = []
    imdb.append({'dataset_name': 'vizwiz', 'version': 1, 'has_answer': False, 'has_gt_layout': False})
    iminfo = {}
    iminfo['image_name'] = im_file.replace('.jpg', '')
    iminfo['img_id'] = int(iminfo['image_name'].split('_')[-1])
    iminfo['question_id'] = 0
    iminfo['feature_path'] = 'COCO_' + iminfo['image_name'].lower() + '.npy'
    iminfo['question_str'] = question_str
    iminfo['question_tokens'] = text_processing.tokenize(iminfo['question_str'])
    imdb.append(iminfo)
    return imdb

def print_result(question_ids, soft_max_result, ans_dic):
    predicted_answers = np.argsort(soft_max_result, axis=1)[0][-3:]
    # predicted_answers = np.argmax(soft_max_result, axis=1)
    answers = []
    scores = []
    for idx, pred_idx in enumerate(predicted_answers):
        pred_ans = ans_dic.idx2word(pred_idx)
        answers.append(pred_ans)
        scores.append(soft_max_result[0][pred_idx])
    return answers, scores

with open(config_file, 'r') as f:
    config = yaml.load(f)

with open(test_annot_file, 'r') as f:
    data = json.load(f)
test_annot = data['annotations']

with open(train_annot_file, 'r') as f:
    data = json.load(f)
train_annot = data['annotations']

with open(val_annot_file, 'r') as f:
    data = json.load(f)
val_annot = data['annotations']

data_set_test = prepare_test_data_set(**config['data'], **config['model'], verbose=True, test_mode=True)
data_set_val = prepare_eval_data_set(**config['data'], **config['model'], verbose=True, test_mode=True)
data_set_train = prepare_train_data_set(**config['data'], **config['model'], verbose=True, test_mode=True)

myModel = build_model(config, data_set_test)
state_dict = torch.load(model_file)['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.','')
    new_state_dict[name] = v

myModel.load_state_dict(new_state_dict)

i = 0

results = '/checkpoint/nvivek/test_7k_answers_copy_only_ocr_partial_score/results/baseline_vizwiz_predict/54278/best_model_predict_test.json'
with open(results, 'r') as f:
    predictions = json.load(f)

im_dir = '/private/home/nvivek/vizwiz/Images'

test_questions_file = '/private/home/nvivek/VQA/training_data/v2_OpenEnded_mscoco_vizwiz_test_questions.json'
val_questions_file = '/private/home/nvivek/VQA/training_data/v2_OpenEnded_mscoco_vizwiz_val_questions.json'

with open(test_questions_file, 'r') as f:
    test_questions = json.load(f)

with open(val_questions_file, 'r') as f:
    val_questions = json.load(f)

val_questions = val_questions['questions']
test_questions = test_questions['questions']

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


val_imdb_file = '/private/home/nvivek/VQA/training_data/imdb/imdb_vizwiz_val_large_ocr_only_att_uns_una_sup_copy.npy'
val_imdb = np.load(val_imdb_file)


source = 'test'
train_index = 0
val_index = 28000
test_index = 20000


ans_dic = data_set_test.answer_dict

total = 0
shuffle(predictions)
modified_pred = []
for prediction in predictions:
    # clear_output()
    im_file = prediction['image']
    im_file_path = os.path.join(im_dir, im_file)
    im_feature_path = os.path.join(im_feat_dir, 'COCO_' + im_file.split('/')[-1].replace(".jpg", ".npy").lower())
    im_id = int(im_file.replace('.jpg','').split('_')[-1])
    print(im_id)
    question = test_questions[im_id - test_index]['question']
    answer = prediction['answer']
    # gt = val_annot[im_id - val_index]['multiple_choice_answer']
    # all_answers = [ans['answer'] for ans in val_annot[im_id - val_index]['answers']]
    data_set_test.datasets[0].imdb = get_imdb(im_file, question)
    data_reader_test = DataLoader(data_set_test, shuffle=False, batch_size=1)
    myModel.eval()
    question_ids, soft_max_result, i_att, it_att = run_model(myModel, data_reader_test, ans_dic.UNK_idx)
    predicted_answers, scores = print_result(question_ids, soft_max_result, ans_dic)
    butd_att = i_att[0].cpu().detach().numpy().reshape((137,)).tolist()
    text_att = it_att[0].cpu().detach().numpy().reshape((137,)).tolist()
    best_text = visualize_att(im_file_path, im_feature_path, text_att, 5)

    total += 1
    print(total)
    if answer == '<copy>':
        new_prediction = {}
        new_prediction['image'] = im_file
        try:
            best_text.encode('ascii')
            new_prediction['answer'] = best_text
        except:
            new_prediction['answer'] = predicted_answers[-2]
        if new_prediction['answer'] == '':
            new_prediction['answer'] = predicted_answers[-2]
        modified_pred.append(new_prediction)
    else:
        modified_pred.append(prediction)

print(len(modified_pred))

with open('copy_only_ocr_partial_score_model_predict_test.json', 'w') as f:
    json.dump(modified_pred, f)
