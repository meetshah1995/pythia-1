import os
import sys
import base64
import csv
import _pickle as cPickle
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True, help="input file")
parser.add_argument("--ocr_infile", type=str, required=True, help="input file")
parser.add_argument("--train_id", type=str, required=False, help="train_id file")
parser.add_argument("--val_id", type=str, required=False, help="val_id file")
parser.add_argument("--out_dir", type=str, required=True, help="imdb output directory")
args = parser.parse_args()

out_dir = args.out_dir


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'classes', 'attributes']
OCR_FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'classes', 'attributes', 'ocr_texts', 'detection_scores', 'recognition_scores']

infile = args.infile
ocr_infile = args.ocr_infile


train_ids_file = args.train_id
val_ids_file = args.val_id

#train_imgids = cPickle.load(open(train_ids_file,'rb'))
#val_imgids = cPickle.load(open(val_ids_file,'rb'))

train_imgids = set(range(0,20000))
val_imgids = set(range(28000,31173))

out_dir = args.out_dir

os.makedirs(os.path.join(out_dir, ''),exist_ok=True)
os.makedirs(os.path.join(out_dir, ''), exist_ok=True)

im_with_ocr_bboxes = 0
max_boxes = -1

print("reading tsv...")
with open(infile, "r") as tsv_in_file:
    with open(ocr_infile, "r") as ocr_tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        ocr_reader = csv.DictReader(ocr_tsv_in_file, delimiter='\t', fieldnames=OCR_FIELDNAMES)
        for item, ocr_item in zip(reader, ocr_reader):
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            classes = item['classes']
            classes = list(json.loads(classes))
            attributes = item['attributes']
            text_feat = []
            for clas, attr in zip(classes, attributes):
                if len(attr) > 0:
                    text_feat.append(attr[0] + ' ' + clas)
                else:
                    text_feat.append(clas)

            ocr_item['num_boxes'] = int(ocr_item['num_boxes'])
            ocr_image_id = int(ocr_item['image_id'])
            ocr_image_w = float(ocr_item['image_w'])
            ocr_image_h = float(ocr_item['image_h'])
            ocr_texts = ocr_item['ocr_texts']
            ocr_texts = json.loads(ocr_texts)

            if (image_id != ocr_image_id):
                print("Image ids did not match")
                print(str(image_id) + " " + str(ocr_image_id))

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
                    text_feat.append(ocr_text)

                im_with_ocr_bboxes += 1

                if (ocr_item['num_boxes'] + item['num_boxes']) > max_boxes:
                    max_boxes = ocr_item['num_boxes'] + item['num_boxes']

                image_bboxes = np.concatenate((image_bboxes, ocr_image_bboxes))
                image_feat = np.concatenate((image_feat, ocr_image_feat))

            assert(len(text_feat) == image_feat.shape[0])
            image_feat_and_boxes = {"image_bboxes": image_bboxes, "image_feat": image_feat, "image_text": text_feat}

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                image_file_name =os.path.join(out_dir,'COCO_vizwiz_train_%012d.npy' % image_id)
                np.save(image_file_name, image_feat_and_boxes)
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                image_file_name =image_file_name =os.path.join(out_dir,'COCO_vizwiz_val_%012d.npy' % image_id)
                np.save(image_file_name, image_feat_and_boxes)
            else:
                assert False, 'Unknown image id: %d' % image_id

print("Images with OCR bbxoes " + str(im_with_ocr_bboxes))
print("Max boxes " + str(max_boxes))
