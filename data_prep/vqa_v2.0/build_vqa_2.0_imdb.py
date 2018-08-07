import numpy as np
import json
import os
from dataset_utils import text_processing
import argparse
from dataset_utils.create_imdb_header import create_header

base_dir = '/private/home/nvivek/VQA/training_data/rcnn_adaptive_vizwiz/vizwiz_att_ans'

with open('attention_sup.json', 'r') as f:
    att_sups = json.load(f)

def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers

def build_imdb(image_set, valid_answer_set, coco_set_name =None, annotation_set_name=None):
    annotation_file = os.path.join(data_dir, 'v2_mscoco_%s_annotations.json')
    question_file = os.path.join(data_dir, 'v2_OpenEnded_mscoco_%s_questions.json')

    print('building imdb %s' % image_set)
    has_answer = False
    has_gt_layout = False
    load_gt_layout = False
    load_answer = False

    annotation_set_name = annotation_set_name if annotation_set_name is not None else image_set

    if os.path.exists(annotation_file % annotation_set_name):
        with open(annotation_file % annotation_set_name) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
        load_answer = True
    '''
    if image_set in ['train2014', 'val2014']:
        load_answer = True
        load_gt_layout = False
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
        #qid2layout_dict = np.load(gt_layout_file % image_set)[()]
    else:
        load_answer = False
        load_gt_layout = False '''

    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = coco_set_name if coco_set_name is not None else image_set.replace('-dev', '')
    image_name_template = 'COCO_' + coco_set_name + '_%012d'
    imdb = [None]*(len(questions)+1)

    unk_ans_count = 0
    uns_count = 0
    copy_count = 0
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        feature_path = image_name + '.npy'
        feat = np.load(os.path.join(base_dir, feature_path))
        image_text = list(feat.item().get('image_text'))
        image_box_source = list(feat.item().get('image_bbox_source'))

        question_str = q['question']
        question_tokens = text_processing.tokenize(question_str)

        iminfo = dict(image_name=image_name,
                      image_id=image_id,
                      question_id=question_id,
                      feature_path=feature_path,
                      question_str=question_str,
                      question_tokens=question_tokens)

        # load answers
        if load_answer:
            copy_score = 0.0
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers

            # for i, text in enumerate(image_text):
            #     if image_box_source[i] == 0:
            #         continue
            #     score = 0.0
            #     if text in valid_answers:
            #         count = valid_answers.count(text)
            #         if count == 1:
            #             score = 1.0/3.0
            #         elif count == 2:
            #             score = 2.0/3.0
            #         else:
            #             score = 1.0
            #     if score > copy_score:
            #         copy_score = score
            #
            # if copy_score > 0.0:
            #     copy_count += 1
            #     valid_answers += ['<copy>']
            #     iminfo['copy_score'] = copy_score

            has_answer = True
            att_sup = att_sups[str(image_id)]
            att_sup += [0.0]*(137-len(att_sup))
            iminfo['att_sup'] = np.asarray(att_sup).reshape(137, 1)
            ans_sup = np.zeros((1, 3))
            if valid_answers.count('unanswerable') >= 3:
                ans_sup[0, 0] = 1
                uns_count += 1
            elif valid_answers.count('unsuitable') >= 3:
                ans_sup[0, 1] = 1
                uns_count += 1
            else:
                ans_sup[0, 2] = 1
            iminfo['ans_sup'] = ans_sup

        if load_gt_layout:
            #gt_layout_tokens = qid2layout_dict[question_id]
            #iminfo['gt_layout_tokens'] = gt_layout_tokens
            has_gt_layout = True

        imdb[n_q+1] = iminfo
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    print('total %d out of %d answers are unanswerable' % (uns_count, len(questions)))
    print('total %d out of %d answers are copyable' % (copy_count, len(questions)))
    header = create_header("vqa",has_answer= has_answer,has_gt_layout=has_gt_layout)
    imdb[0] = header
    return imdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    parser.add_argument("--out_dir", type=str, required=True, help="imdb output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    vocab_answer_file = os.path.join(data_dir, 'answers_vizwiz_7k_no_copy.txt')

    answer_dict = text_processing.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)
    print(len(valid_answer_set))

    imdb_vizwiz_train = build_imdb('vizwiz_train_aug', valid_answer_set, coco_set_name='vizwiz_train')
    imdb_vizwiz_val = build_imdb('vizwiz_val_aug', valid_answer_set, coco_set_name='vizwiz_val')
    imdb_vizwiz_test = build_imdb('vizwiz_test', valid_answer_set)

    imdb_dir = os.path.join(out_dir, 'imdb')
    os.makedirs(imdb_dir, exist_ok=True)
    np.save(os.path.join(imdb_dir, 'imdb_vizwiz_train_7k_att_ans_aug.npy'), np.array(imdb_vizwiz_train))
    np.save(os.path.join(imdb_dir, 'imdb_vizwiz_val_7k_att_ans_aug.npy'), np.array(imdb_vizwiz_val))
    np.save(os.path.join(imdb_dir, 'imdb_vizwiz_test_7k_att_ans.npy'), np.array(imdb_vizwiz_test))
