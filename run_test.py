import yaml
import argparse
from torch.utils.data import DataLoader
from train_model.dataset_utils import prepare_test_data_set,prepare_eval_data_set
import torch
from train_model.helper import run_model, print_result, build_model
from collections import OrderedDict
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="base dir")
    # parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="output file name prefix, will append .json or .pkl")
    # parser.add_argument("--model_path", type=str, help="path of model", required=True)
    parser.add_argument("--batch_size", type=int,
                        help="batch_size for test, o.w. using the one in config file", default=None)
    parser.add_argument("--num_workers",type=int, help="num_workers in dataLoader, default 0", default=5)
    parser.add_argument("--json_only", action='store_true', help="flag for only need json result")
    parser.add_argument("--use_val",action='store_true',help="flag for using val data for test")


    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    args = parse_args()

    base_dir = args.base_dir
    config_file = os.path.join(base_dir, 'config.yaml')
    out_prefix = os.path.join(base_dir, args.out_prefix)
    out_file = out_prefix + ".json"
    model_file = os.path.join(base_dir, 'best_model.pth')

    with open(config_file, 'r') as f:
        config = yaml.load(f)

    batch_size = config['data']['batch_size'] if args.batch_size is None else args.batch_size
    if args.use_val:
        data_set_test = prepare_eval_data_set(**config['data'], **config['model'], verbose=True)
    else:
        data_set_test = prepare_test_data_set(**config['data'], **config['model'], verbose=True)
    data_reader_test = DataLoader(data_set_test, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
    ans_dic = data_set_test.answer_dict

    myModel = build_model(config, data_set_test)
    state_dict = torch.load(model_file)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','')
        new_state_dict[name] = v
    myModel.load_state_dict(new_state_dict)

    question_ids, soft_max_result, _, _ = run_model(myModel, data_reader_test, ans_dic.UNK_idx)

    pkl_res_file = out_prefix + ".pkl" if not args.json_only else None

    print(pkl_res_file)

    test = "test"
    if args.use_val:
        test = "val"

    print(args.use_val)
    print(test)
    print_result(question_ids, soft_max_result, ans_dic, out_file, args.json_only, pkl_res_file, test)
