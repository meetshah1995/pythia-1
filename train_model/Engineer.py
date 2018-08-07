import shutil
import torch
import torch.nn as nn
import sys
import os
from global_variables.global_variables import use_cuda
import timeit
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from config.config import cfg
import gc
import operator as op
import functools

device = torch.device("cuda" if use_cuda else "cpu")


def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots = one_hots.cuda() if use_cuda else one_hots
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def one_stage_train(myModel, data_reader_trn, myOptimizer,
                    loss_criterion, snapshot_dir, log_dir,
                    i_iter, start_epoch, data_reader_eval=None,
                    scheduler=None, use_attention_supervision=False,
                    use_answer_supervision=False,
                    att_loss_criterion=None,
                    ans_loss_criterion=None,
                    att_loss_weight=None,
                    ans_loss_weight=None):
    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter

    avg_accuracy = 0
    accuracy_decay = 0.99
    best_val_accuracy = 0
    writer = SummaryWriter(log_dir)
    print(log_dir)
    best_iter = i_iter
    iepoch = start_epoch
    start = timeit.default_timer()
    while i_iter < max_iter:
        n_sample_tot = 0

        start_iter = timeit.default_timer()
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):

            i_iter += 1
            if i_iter > max_iter:
                break

            scheduler.step(i_iter)

            answer_scores = batch['ans_scores']
            n_sample = answer_scores.size(0)
            n_sample_tot += n_sample
            myOptimizer.zero_grad()

            add_graph = False
            logit_res, i_att, it_att, ans_res = one_stage_run_model(batch,
                                                           myModel,
                                                           add_graph,
                                                           log_dir)

            input_answers_variable = answer_scores.type(
                torch.FloatTensor).to(device)

            total_loss = loss_criterion(logit_res, input_answers_variable)
            att_loss = None
            ans_loss = None
            if use_attention_supervision:
                att_sup = batch['att_sup']
                att_sup_variable = att_sup.type(torch.FloatTensor).to(device)
                # F.normalize(att_sup_variable, p=1, dim=1)
                att_loss = att_loss_criterion(it_att[0], att_sup_variable)
                # att_loss_2 = att_loss_criterion(i_att[0], att_sup_variable)
                # total_loss = sum([total_loss, att_loss])

            if use_answer_supervision:
                ans_sup = batch['ans_sup']
                ans_sup_variable = ans_sup.type(torch.FloatTensor).to(device)
                ans_loss = ans_loss_criterion(ans_res, ans_sup_variable)
                # total_loss = sum([total_loss, ans_loss])

            if use_answer_supervision and use_attention_supervision:
                total_loss_weight = 1 - att_loss_weight - ans_loss_weight
                total_loss = (total_loss_weight*total_loss) + (att_loss_weight*att_loss) + (ans_loss_weight*ans_loss)
            elif use_answer_supervision:
                total_loss = (1 - ans_loss_weight)*total_loss + (ans_loss_weight*ans_loss)
            elif use_attention_supervision:
                total_loss = (1 - att_loss_weight)*total_loss + (att_loss_weight*att_loss)

            total_loss.backward()

            if max_grad_l2_norm is not None:
                if clip_norm_mode == 'all':
                    norm = nn.utils.clip_grad_norm(
                        myModel.parameters(), max_grad_l2_norm)
                    writer.add_scalar('grad_norm', norm, i_iter)
                elif clip_norm_mode == 'question':
                    norm = nn.utils.clip_grad_norm(
                        myModel.module.question_embedding_models.parameters(),
                        max_grad_l2_norm)
                    writer.add_scalar('question_grad_norm', norm, i_iter)
                else:
                    raise NotImplementedError

            for param in myModel.module.image_text_feat_embedding_models_list.named_parameters():
                writer.add_scalar("TF" + param[0], param[1].grad.norm(), i_iter)
            for param in myModel.module.image_embedding_models_list.named_parameters():
                writer.add_scalar("IF" + param[0], param[1].grad.norm(), i_iter)
            for param in myModel.module.question_embedding_models.named_parameters():
                writer.add_scalar(param[0], param[1].grad.norm(), i_iter)

            myOptimizer.step()

            scores = torch.sum(
                compute_score_with_logits(
                    logit_res, input_answers_variable.data))
            accuracy = scores / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            if i_iter % report_interval == 0:
                cur_loss = total_loss.data[0]
                if use_attention_supervision:
                    cur_att_loss = att_loss.data[0]
                else:
                    cur_att_loss = -100.0
                if use_answer_supervision:
                    cur_ans_loss = ans_loss.data[0]
                else:
                    cur_ans_loss = -100.0
                end_iter = timeit.default_timer()
                time = end_iter - start_iter
                start_iter = timeit.default_timer()
                val_batch = next(iter(data_reader_eval))
                val_score, val_loss = evaluate_a_batch(val_batch,
                                                       myModel,
                                                       loss_criterion)

                print("iter:", i_iter, "train_loss: %.4f" % cur_loss,
                      " train_score: %.4f" % accuracy,
                      " avg_train_score: %.4f" % avg_accuracy,
                      "val_score: %.4f" % val_score,
                      "val_loss: %.4f" % val_loss,
                      "att_loss: %.4f" % cur_att_loss,
                      "ans_loss: %.4f" % cur_ans_loss,
                      "time(s): %.1f" % time)
                sys.stdout.flush()

                writer.add_scalar('train_loss', cur_loss, i_iter)
                writer.add_scalar('train_score', accuracy, i_iter)
                writer.add_scalar('train_score_avg', avg_accuracy, i_iter)
                writer.add_scalar('val_score', val_score, i_iter)
                writer.add_scalar('val_loss', val_loss, i_iter)
                writer.add_scalar('att_loss', cur_att_loss, i_iter)
                writer.add_scalar('ans_loss', cur_ans_loss, i_iter)
                # write out parameters
                for name, param in myModel.named_parameters():
                    writer.add_histogram(name,
                                         param.clone().cpu().data.numpy(),
                                         i_iter)

            if i_iter % snapshot_interval == 0 or i_iter == max_iter:
                # evaluate the model when finishing one epoch
                if data_reader_eval is not None:
                    val_accuracy, upbound_acc, val_sample_tot = (
                        one_stage_eval_model(data_reader_eval, myModel))
                    end = timeit.default_timer()
                    epoch_time = end - start
                    start = timeit.default_timer()
                    print("i_epoch:", iepoch,
                          "i_iter:", i_iter,
                          "val_acc:%.4f" % val_accuracy,
                          "runtime(s):%d" % epoch_time)
                    sys.stdout.flush()

                model_snapshot_file = os.path.join(snapshot_dir,
                                                   "model_%08d.pth" % i_iter)
                model_result_file = os.path.join(snapshot_dir,
                                                 "result_on_val.txt")
                torch.save({
                    'epoch': iepoch,
                    'iter': i_iter,
                    'state_dict': myModel.state_dict(),
                    'optimizer': myOptimizer.state_dict(),
                }, model_snapshot_file)
                with open(model_result_file, 'a') as fid:
                    fid.write('%d %d %.5f\n' % (iepoch,
                                                i_iter, val_accuracy * 100))

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = iepoch
                    best_iter = i_iter
                    best_model_snapshot_file = os.path.join(snapshot_dir,
                                                            "best_model.pth")
                    shutil.copy(model_snapshot_file, best_model_snapshot_file)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print("best_acc:%.6f after epoch: %d/%d at iter %d" %
          (best_val_accuracy, best_epoch, iepoch, best_iter))
    sys.stdout.flush()

def evaluate_a_batch(batch, myModel, loss_criterion):
    myModel.eval()
    answer_scores = batch['ans_scores']
    n_sample = answer_scores.size(0)
    input_answers_variable = answer_scores.type(torch.FloatTensor).to(device)

    logit_res, i_att, it_att, _ = one_stage_run_model(batch, myModel)
    predicted_scores = torch.sum(
        compute_score_with_logits(logit_res, input_answers_variable.data))
    total_loss = loss_criterion(logit_res, input_answers_variable)

    myModel.train()
    return predicted_scores / n_sample, total_loss.data[0]


def one_stage_eval_model(data_reader_eval, myModel):
    val_score_tot = 0
    val_sample_tot = 0
    upbound_tot = 0
    myModel.eval()
    for idx, batch in enumerate(data_reader_eval):
        answer_scores = batch['ans_scores']
        n_sample = answer_scores.size(0)
        answer_scores = answer_scores.cuda() if use_cuda else answer_scores
        logit_res, i_att, it_att, _ = one_stage_run_model(batch, myModel)
        predicted_scores = torch.sum(
            compute_score_with_logits(logit_res, answer_scores))
        upbound = torch.sum(torch.max(answer_scores, dim=1)[0])

        val_score_tot += predicted_scores
        val_sample_tot += n_sample
        upbound_tot += upbound

    myModel.train()
    return (val_score_tot / val_sample_tot,
            upbound_tot / val_sample_tot, val_sample_tot)



def one_stage_run_model(batch, myModel, add_graph=False, log_dir=None):

    input_text_seqs = batch['input_seq_batch']
    input_images = batch['image_feat_batch']
    image_text_feat = batch.get('image_text_vector')

    input_txt_variable = input_text_seqs.type(torch.LongTensor).to(device)

    image_feat_variable = input_images.to(device)
    image_feat_variables = [image_feat_variable]

    # Support only one image text feat variable
    image_text_feat_variables = None
    if image_text_feat is not None:
        # image_text_feat_variable = image_text_feat.type(
        #     torch.LongTensor).to(device)
        image_text_feat_variable = image_text_feat.to(device)
        image_text_feat_variables = [image_text_feat_variable]

    image_dim_variable = None
    if 'image_dim' in batch:
        image_dim_variable = batch['image_dim'].to(device)

    # check if more than 1 image_feat_batch
    i = 1
    image_feat_key = "image_feat_batch_%s"
    while image_feat_key % str(i) in batch:
        image_feat = batch[image_feat_key % str(i)]
        if type(image_feat) is list:
            image_feat = image_feat[0]
        tmp_image_variable = image_feat.to(device)
        image_feat_variables.append(tmp_image_variable)
        i += 1

    logit_res, i_att, it_att, ans_res = myModel(
        input_question_variable=input_txt_variable,
        image_dim_variable=image_dim_variable,
        image_feat_variables=image_feat_variables,
        image_text_feat_variables=image_text_feat_variables)

    if add_graph:
        with SummaryWriter(log_dir=log_dir, comment='basicblock') as w:
            w.add_graph(myModel, (input_txt_variable,
                                  image_dim_variable,
                                  image_feat_variables))

    return logit_res, i_att, it_att, ans_res
