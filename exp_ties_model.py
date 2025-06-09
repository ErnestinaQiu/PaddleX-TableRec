"""
exp
author: Ernestina Qiu
"""
import os
import gc
import yaml
import logging
import numpy as np
from datetime import datetime

import paddle
import paddle.optimizer as optim
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

    cell_prob_adj_upper_tri = paddle.tensor.triu(cell_prob_adj_mat, diagonal=1)
    cell_pred_adj_upper_tri = paddle.tensor.triu(cell_pred_adj_mat, diagonal=1)
    row_prob_adj_upper_tri = paddle.tensor.triu(row_prob_adj_mat, diagonal=1)
    row_pred_adj_upper_tri = paddle.tensor.triu(row_pred_adj_mat, diagonal=1)
    col_prob_adj_upper_tri = paddle.tensor.triu(col_prob_adj_mat, diagonal=1)
    col_pred_adj_upper_tri = paddle.tensor.triu(col_pred_adj_mat, diagonal=1)

    y_cell_adj_mats_upper_tri = paddle.tensor.triu(y_cell_adj_mats, diagonal=1)
    y_row_adj_mats_upper_tri = paddle.tensor.triu(y_row_adj_mats, diagonal=1)
    y_col_adj_mats_upper_tri = paddle.tensor.triu(y_col_adj_mats, diagonal=1)

    correct_cell = paddle.equal(x=cell_pred_adj_upper_tri, y=y_cell_adj_mats_upper_tri).astype(paddle.float32)
    cell_acc = paddle.mean(correct_cell)
    cell_loss = F.binary_cross_entropy(input=cell_prob_adj_upper_tri, label=y_cell_adj_mats_upper_tri)

    correct_row = paddle.equal(x=row_pred_adj_upper_tri, y=y_row_adj_mats_upper_tri).astype(paddle.float32)
    row_acc = paddle.mean(correct_row)
    row_loss = F.binary_cross_entropy(input=row_prob_adj_upper_tri, label=y_row_adj_mats_upper_tri)

    correct_col = paddle.equal(x=col_pred_adj_upper_tri, y=y_col_adj_mats_upper_tri).astype(paddle.float32)
    col_acc = paddle.mean(correct_col)
    col_loss = F.binary_cross_entropy(input=col_prob_adj_upper_tri, label=y_col_adj_mats_upper_tri)

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
    learning_rate = config['learning_rate']

    val_acc_history = []
    val_loss_history = []

    paddle.set_device('gpu')

    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")

    save_dir = f'./output/exp/TIES/{date_string}'

    ds_logger = get_logger(name='exp_ds', log_file=os.path.join(save_dir, 'exp_ds.log'), log_level=logging.CRITICAL)
    train_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='train')
    val_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='val')

    md_logger = get_logger(name='exp_md', log_file=os.path.join(save_dir, f'exp_same_row_{date_string}.log'), log_level=logging.DEBUG)
    model = BasicModel(config, logger=md_logger)

    md_save_dir = os.path.join(save_dir, 'models')
    os.makedirs(md_save_dir, exist_ok=True)
    best_md_save_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(best_md_save_dir, exist_ok=True)
    best_md_sp = os.path.join(best_md_save_dir, 'best_model.pdparams')

    pretrained_model_path = './output/exp/TIES/20250606120132/best_model/best_model.pdparams'
    if os.path.exists(pretrained_model_path):
        model.set_state_dict(paddle.load(pretrained_model_path))
        print(f"Pretrained model loaded from {pretrained_model_path}")
    else:
        print(f"No pretrained model found at {pretrained_model_path}. Training from scratch.")

    model.train()
    opt = paddle.optimizer.Adam(
        learning_rate=learning_rate, parameters=model.parameters()
    )

    scheduler = optim.lr.ReduceOnPlateau(learning_rate=learning_rate, mode='min', factor=0.5, patience=10, threshold=1e-4, cooldown=3, verbose=True)

    md_logger.info('start training ...')

    best_acc = 0.0
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

            scheduler.step(metrics=train_loss)
            current_lr = scheduler.last_lr
            md_logger.info(f"[train] epoch: {epoch}, batch_id: {batch_id}, train_acc is: {train_acc}, loss is: {train_loss.numpy()}, lr: {current_lr}\ntrain_cell_loss: {train_cell_loss}, train_cell_acc: {train_cell_acc}, train_row_loss: {train_row_loss}, train_row_acc: {train_row_acc}, train_col_loss: {train_col_loss}, train_col_acc: {train_col_acc}")
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(md_save_dir, f"model_epoch_{epoch+1}_row_acc_{train_row_acc:.4f}.pdparams")
                paddle.save(model.state_dict(), save_path)

            if train_acc > best_acc:
                best_acc = train_acc
                paddle.save(model.state_dict(), best_md_sp)
                md_logger.info(f'Best model saved at {best_md_sp}')

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
        md_logger.info(f"[validation] acc/loss: {val_acc.numpy()}/{val_loss.numpy()}, cell_loss is: {val_cell_loss}, row_loss: {val_row_loss}, col_loss: {val_col_loss}/ncell_acc: {val_cell_acc}, row_acc: {val_row_acc}, col_acc: {val_col_acc}")
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()
        del train_d
        del val_d
        gc.collect()
        if (epoch + 1) % 2 == 0:
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

if __name__ == "__main__":
    exp()
