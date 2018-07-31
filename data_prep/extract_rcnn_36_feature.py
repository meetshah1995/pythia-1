import os
import sys
import base64
import csv
import _pickle as cPickle
import numpy as np
import argparse
import json
from dataset_utils import text_processing
import codecs
import torch

def get_overlap_scores(a, b):
    if len(a) < len(b):
        c = a
        a = b
        b = c
    overlap = 0.0
    while len(b) >= 2:
        if b in a:
            overlap = len(b)
            return overlap*1.0/len(a)
        else:
            b = b[:-1]
    return 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True, help="input file")
parser.add_argument("--ocr_infile", type=str, required=True, help="input file")
parser.add_argument("--train_id", type=str, required=False, help="train_id file")
parser.add_argument("--val_id", type=str, required=False, help="val_id file")
parser.add_argument("--out_dir", type=str, required=True, help="imdb output directory")
args = parser.parse_args()

out_dir = args.out_dir

with open('../training_data/v2_mscoco_vizwiz_train_annotations.json', 'r') as f:
    train_annot = json.load(f)
with open('../training_data/v2_mscoco_vizwiz_val_annotations.json', 'r') as f:
    val_annot = json.load(f)

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'classes', 'attributes']
OCR_FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'classes', 'attributes', 'ocr_texts', 'detection_scores', 'recognition_scores']

infile = args.infile
ocr_infile = args.ocr_infile

embed_dim = 300

box_vectors_file = codecs.open('box_text_vectors_wiki_en', 'r', 'utf-8')

box_vectors = {}

for line in box_vectors_file:
    data = line.rstrip().split()
    embed = data[-embed_dim:]
    embed = [float(x) for x in embed]
    text_len = len(data) - embed_dim
    text = ' '.join(data[0:text_len])
    box_vectors[text] = embed

train_ids_file = args.train_id
val_ids_file = args.val_id

train_annot = train_annot['annotations']
val_annot = val_annot['annotations']

#train_imgids = cPickle.load(open(train_ids_file,'rb'))
#val_imgids = cPickle.load(open(val_ids_file,'rb'))

train_imgids = set(range(0, 20000))
test_imgids = set(range(20000, 28000))
val_imgids = set(range(28000, 31173))

out_dir = args.out_dir

att_sups = {}

os.makedirs(os.path.join(out_dir, ''),exist_ok=True)
os.makedirs(os.path.join(out_dir, ''), exist_ok=True)

im_with_ocr_bboxes = 0
max_boxes = -1
min_boxes = 300
train_index = 0
val_index = 28000
print("reading tsv...")
with open(infile, "r") as tsv_in_file:
    with open(ocr_infile, "r") as ocr_tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        ocr_reader = csv.DictReader(ocr_tsv_in_file, delimiter='\t', fieldnames=OCR_FIELDNAMES)
        for item, ocr_item in zip(reader, ocr_reader):
            box_sentence_vectors = np.empty((0, embed_dim), float)
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            classes = item['classes']
            classes = list(json.loads(classes))
            attributes = list(json.loads(item['attributes']))
            text_feat = []
            att_sup = []
            gt = ''
            if image_id in train_imgids:
                gt = train_annot[int(image_id) - train_index]['multiple_choice_answer']
            elif image_id in val_imgids:
                gt = val_annot[int(image_id) - val_index]['multiple_choice_answer']

            gt_tokens = gt.split()

            for clas, attr in zip(classes, attributes):
                text = ''
                clas = ' '.join(clas.split(','))
                if len(attr) > 0:
                    text_feat.append(attr[0] + ' ' + clas)
                    text = attr[0] + ' ' + clas
                else:
                    text_feat.append(clas)
                    text = clas
                text_tokens = text_processing.tokenize(text)
                to_append = 0.0
                for token in text_tokens:
                    if token in gt_tokens:
                        to_append = 1.0
                        # print(token)
                        # print(gt_tokens)
                        break
                to_append = get_overlap_scores(
                    ' '.join(text_tokens), ' '.join(gt_tokens))
                # print(' '.join(text_tokens) + " | " + ' '.join(gt_tokens) + ' | ' + str(to_append))
                att_sup.append(to_append)
                box_sentence_vector = box_vectors[text]
                box_sentence_vector = np.asarray(box_sentence_vector, dtype = float)
                box_sentence_vector = np.reshape(box_sentence_vector, (1, embed_dim))
                box_sentence_vectors = np.append(
                    box_sentence_vectors, box_sentence_vector, axis=0)

            image_bbox_source = np.zeros((item['num_boxes'], 1), dtype=float)
            ocr_item['num_boxes'] = int(ocr_item['num_boxes'])
            ocr_image_id = int(ocr_item['image_id'])
            ocr_image_w = float(ocr_item['image_w'])
            ocr_image_h = float(ocr_item['image_h'])
            ocr_texts = ocr_item['ocr_texts']
            ocr_texts = json.loads(ocr_texts)

            if (image_id != ocr_image_id):
                print("Image ids did not match")
                print(str(image_id) + " " + str(ocr_image_id))
                break

            if (item['num_boxes']) > max_boxes:
                max_boxes = item['num_boxes']

            image_bboxes = np.frombuffer(
                base64.b64decode(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            image_feat = np.frombuffer(
                base64.b64decode(item['features']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            if ocr_item['num_boxes'] > 0:
                ocr_image_bboxes = np.frombuffer(
                    base64.b64decode(ocr_item['boxes']),
                    dtype=np.float32).reshape((ocr_item['num_boxes'], -1))

                ocr_image_feat = np.frombuffer(
                    base64.b64decode(ocr_item['features']),
                    dtype=np.float32).reshape((ocr_item['num_boxes'], -1))

                for ocr_text in ocr_texts:
                    text_feat.append(ocr_text.lower())

                    box_sentence_vector = box_vectors[ocr_text]
                    box_sentence_vector = np.asarray(box_sentence_vector, dtype = float)
                    box_sentence_vector = np.reshape(box_sentence_vector, (1, embed_dim))
                    box_sentence_vectors = np.append(
                        box_sentence_vectors, box_sentence_vector, axis=0)


                    ocr_text_tokens = text_processing.tokenize(ocr_text)
                    to_append = 0.0
                    for token in ocr_text_tokens:
                        if token in gt_tokens:
                            # print(token)
                            # print(gt_tokens)
                            to_append = 1.0
                            break

                    to_append = get_overlap_scores(
                        ' '.join(ocr_text_tokens), ' '.join(gt_tokens))
                    # print(' '.join(ocr_text_tokens) + " | " + ' '.join(gt_tokens) + ' | ' + str(to_append))
                    att_sup.append(to_append)

                im_with_ocr_bboxes += 1

                if (ocr_item['num_boxes'] + item['num_boxes']) > max_boxes:
                    max_boxes = ocr_item['num_boxes'] + item['num_boxes']

                image_bboxes = np.concatenate((image_bboxes, ocr_image_bboxes))
                image_feat = np.concatenate((image_feat, ocr_image_feat))
                ocr_boxes = np.ones((ocr_item['num_boxes'], 1), dtype=float)
                image_bbox_source = np.concatenate((image_bbox_source, ocr_boxes), axis=0)

            if image_id in test_imgids:
                att_sup = None

            text_feat = np.asanyarray(text_feat)
            if (image_feat.shape[0] < 10):
                print(image_id)
                print(item['num_boxes'])
                print(ocr_item['num_boxes'])
            assert(text_feat.shape[0] == image_feat.shape[0])
            assert(box_sentence_vectors.shape[0] == image_feat.shape[0])
            assert(text_feat.shape[0] == image_bbox_source.shape[0])
            image_feat_and_boxes = {"image_bboxes": image_bboxes,
                                    "image_feat": image_feat,
                                    "image_text": text_feat,
                                    "image_bbox_source": image_bbox_source,
                                    "image_text_vector": box_sentence_vectors}

            if att_sup is not None:
                assert(len(att_sup) == image_feat.shape[0])
                att_sups[image_id] = att_sup
            # image_feat_and_boxes = {"image_bboxes": image_bboxes, "image_feat": image_feat}

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                image_file_name =os.path.join(out_dir,'COCO_vizwiz_train_%012d.npy' % image_id)
                np.save(image_file_name, image_feat_and_boxes)
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                image_file_name =image_file_name =os.path.join(out_dir,'COCO_vizwiz_val_%012d.npy' % image_id)
                np.save(image_file_name, image_feat_and_boxes)
            elif image_id in test_imgids:
                test_imgids.remove(image_id)
                image_file_name =image_file_name =os.path.join(out_dir,'COCO_vizwiz_test_%012d.npy' % image_id)
                np.save(image_file_name, image_feat_and_boxes)
            else:
                assert False, 'Unknown image id: %d' % image_id

print("Images with OCR bbxoes " + str(im_with_ocr_bboxes))
print("Max boxes " + str(max_boxes))

with open('attention_sup.json', 'w') as f:
    json.dump(att_sups, f)
