import json
import pickle
import argparse
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    parser.add_argument("--mode", type=str, required=True, help="imdb output directory")
    args = parser.parse_args()
    data_dir = args.data_dir
    mode = args.mode

    ifh = os.path.join(data_dir, mode + '.json')

    with open(ifh, 'r') as f:
        data = json.load(f)

    new_data = {}
    new_data['annotations'] = []
    questions = {}
    questions['questions'] = []
    question_id = 0
    ans = set()
    imgids = set()
    for annot in data:
        question_info = {}
        question = annot['question']
        image = annot['image']
        image_id = int(image.split('.')[0].split('_')[-1])
        imgids.add(image_id)
        annot['image_id'] = image_id
        if mode != 'test':
            answers = annot['answers']
            answer_id = 0
            answer_count = {}
            for answer in answers:
                if (answer['answer'] in answer_count.keys()):
                    answer_count[answer['answer']] = answer_count[answer['answer']] + 1
                else:
                    answer_count[answer['answer']] = 1
                answer['answer_id'] = answer_id
                answer_id = answer_id + 1
                ans.add(answer['answer'])
            a1_sorted_keys = sorted(answer_count, key=answer_count.get, reverse=True)
            annot['multiple_choice_answer'] = a1_sorted_keys[0]
            annot['answers'] = answers

        question_info['question'] = question
        question_info['question_id'] = question_id
        question_info['image_id'] = image_id
        annot['question_id'] = question_id
        question_id = question_id + 1
        questions['questions'].append(question_info)
        new_data['annotations'].append(annot)

    qfh = ('v2_OpenEnded_mscoco_vizwiz_%s_questions.json' % mode)
    afh = ('v2_mscoco_vizwiz_%s_annotations.json' % mode)
    idsfh = ('%sids.pkl' % mode)

    print(len(ans))

    with open(idsfh, 'wb') as f:
        pickle.dump(imgids, f)

    with open(qfh, 'w') as f:
        json.dump(questions, f)

    with open(afh, 'w') as f:
        json.dump(new_data, f)
