"""
exp
author: Ernestina Qiu
"""
import os
import yaml
import logging
import numpy as np

import paddle
import paddle.nn.functional as F
from exp.model.ties.models.basic_model import BasicModel
from exp.data.ties_dataset import TiesDataSet
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


def get_pred_detail(d:dict, model, config:dict):
    images = d['images']
    cell_boxes = d['cell_boxes']
    ocr_res_boxes = d['ocr_res_boxes']
    y_cell_adj_mats = d['cell_adj_mats']
    y_row_adj_mats = d['row_adj_mats']
    y_col_adj_mats = d['col_adj_mats']
    x_data = {"images": images, "text_boxes": ocr_res_boxes}

    pred_dict = model(x_data)
    cell_prob_adj_mat = pred_dict['cell_prob_adj_matrix']
    cell_pred_adj_mat = pred_dict['cell_pred_adj_matrix']
    row_prob_adj_mat = pred_dict['row_prob_adj_matrix']
    row_pred_adj_mat = pred_dict['row_pred_adj_matrix']
    col_prob_adj_mat = pred_dict['col_prob_adj_matrix']
    col_pred_adj_mat = pred_dict['col_pred_adj_matrix']

    correct_cell = paddle.equal(x=cell_pred_adj_mat, y=y_cell_adj_mats).astype(paddle.float32)
    cell_acc = paddle.mean(correct_cell)
    cell_loss = F.binary_cross_entropy(input=cell_pred_adj_mat, label=y_cell_adj_mats)

    correct_row = paddle.equal(x=row_pred_adj_mat, y=y_row_adj_mats).astype(paddle.float32)
    row_acc = paddle.mean(correct_row)
    row_loss = F.binary_cross_entropy(input=row_pred_adj_mat, label=y_row_adj_mats)

    correct_col = paddle.equal(x=col_pred_adj_mat, y=y_col_adj_mats).astype(paddle.float32)
    col_acc = paddle.mean(correct_col)
    col_loss = F.binary_cross_entropy(input=col_pred_adj_mat, label=y_col_adj_mats)

    _loss_cell_weight = config['loss_cell_weight']
    _loss_row_weight = config['loss_row_weight']
    _loss_col_weight = config['loss_col_weight']

    loss_cell_weight = _loss_cell_weight / (_loss_cell_weight + _loss_row_weight + _loss_col_weight)
    loss_row_weight = _loss_row_weight / (_loss_cell_weight + _loss_row_weight + _loss_col_weight)
    loss_col_weight = _loss_col_weight / (_loss_cell_weight + _loss_row_weight + _loss_col_weight)

    weighted_loss = loss_cell_weight * cell_loss + loss_row_weight * row_loss + loss_col_weight * col_loss
    weighted_acc = loss_cell_weight * cell_acc + loss_row_weight * row_acc + loss_col_weight * col_acc

    return {"cell_acc": cell_acc, "cell_loss": cell_loss, 'row_acc': row_acc, 'row_loss': row_loss, 'col_acc': col_acc, 'col_loss': col_loss, 'weighted_loss': weighted_loss, 'weighted_acc': weighted_acc}


def exp():
    config_path = os.path.join(os.getcwd(), 'exp/configs/ties.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    epoch_num = config['epoch_num']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    val_acc_history = []
    val_loss_history = []

    paddle.set_device('gpu')

    ds_logger = get_logger(name='exp_ds', log_file='./output/tmp/exp_ds.log', log_level=logging.CRITICAL)
    train_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='train', seed=123)
    val_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='val', seed=123)

    md_logger = get_logger(name='exp_md', log_file='./output/tmp/exp.log', log_level=logging.DEBUG)
    model = BasicModel(config, logger=md_logger)

    model.train()
    opt = paddle.optimizer.Adam(
        learning_rate=learning_rate, parameters=model.parameters()
    )

    print('start training ...')

    for epoch in range(epoch_num):
        for batch_id, train_d in enumerate(train_ties_ds):
            train_detail = get_pred_detail(train_d, model=model, config=config)
            train_loss = train_detail['weighted_loss']
            train_acc = train_detail['weighted_acc']
            train_cell_loss = train_detail['cell_loss']
            train_cell_acc = train_detail['cell_acc']
            train_row_loss = train_detail['row_loss']
            train_row_acc = train_detail['row_acc']
            train_col_loss = train_detail['col_loss']
            train_col_acc = train_detail['col_acc']

            if batch_id % 1 == 0:
                print(f"epoch: {epoch}, batch_id: {batch_id}, train_acc is: {train_acc}, loss is: {train_loss.numpy()}, train_cell_loss is: {train_cell_loss}, train_row_loss: {train_row_loss}, train_col_loss: {train_col_loss}\ntrain_cell_acc: {train_cell_acc}, train_row_acc: {train_row_acc}, train_col_acc: {train_col_acc}")

            train_loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, val_d in enumerate(val_ties_ds):
            val_detail = get_pred_detail(val_d, model=model, config=config)
            val_loss = val_detail['weighted_loss']
            val_acc = val_detail['weighted_acc']
            val_cell_loss = val_detail['cell_loss']
            val_cell_acc = val_detail['cell_acc']
            val_row_loss = val_detail['row_loss']
            val_row_acc = val_detail['row_acc']
            val_col_loss = val_detail['col_loss']
            val_col_acc = val_detail['col_acc']

            accuracies.append(val_acc.numpy())
            losses.append(val_loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print(f"[validation] acc/loss: {val_acc.numpy()}/{val_loss.numpy()}, cell_loss is: {val_cell_loss}, row_loss: {val_row_loss}, col_loss: {val_col_loss}\ncell_acc: {val_cell_acc}, row_acc: {val_row_acc}, col_acc: {val_col_acc}")
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()

if __name__ == "__main__":
    exp()
