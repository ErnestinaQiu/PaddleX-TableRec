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
    row_prob_adj_mat = pred_dict['row_prob_adj_matrix']
    row_prob_adj_upper_tri = paddle.tensor.triu(row_prob_adj_mat, diagonal=1)

    row_pred_adj_mat = pred_dict['row_pred_adj_matrix']
    row_pred_adj_upper_tri = paddle.tensor.triu(row_pred_adj_mat, diagonal=1)

    y_row_adj_upper_tri = paddle.tensor.triu(y_row_adj_mats, diagonal=1)

    correct_row = paddle.equal(x=row_pred_adj_upper_tri, y=y_row_adj_upper_tri).astype(paddle.float32)
    row_acc = paddle.mean(correct_row)
    row_loss = F.binary_cross_entropy(input=row_prob_adj_upper_tri, label=y_row_adj_upper_tri)

    return {'row_acc': row_acc, 'row_loss': row_loss}


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
    train_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='train', seed=123)
    val_ties_ds = TiesDataSet(config=config, logger=ds_logger, mode='val', seed=123)

    md_logger = get_logger(name='exp_md', log_file=os.path.join(save_dir, f'exp_same_row_{date_string}.log'), log_level=logging.DEBUG)
    model = BasicModel(config, logger=md_logger)

    md_save_dir = os.path.join(save_dir, 'models')
    os.makedirs(md_save_dir, exist_ok=True)
    best_md_save_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(best_md_save_dir, exist_ok=True)
    best_md_sp = os.path.join(best_md_save_dir, 'best_model.pdparams')

    pretrained_model_path = './output/exp/TIES/20250603210019/best_model/best_model.pdparams'
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

    best_row_acc = 0.0
    for epoch in range(epoch_num):
        for batch_id, train_d in enumerate(train_ties_ds):
            train_detail = get_pred_detail(train_d, model=model, config=config)
            train_loss = train_detail['row_loss']
            train_acc = train_detail['row_acc']

            scheduler.step(metrics=train_loss)

            md_logger.info(f"[train] epoch: {epoch}, batch_id: {batch_id}, train_acc is: {train_acc}, loss is: {train_loss.numpy()}, train_loss: {train_loss}, train_acc: {train_acc}")

            if (batch_id + 1) % 5 == 0:
                save_path = os.path.join(md_save_dir, f"model_epoch_{epoch+1}_row_acc_{train_acc:.4f}.pdparams")
                paddle.save(model.state_dict(), save_path)

            if train_acc > best_row_acc:
                best_row_acc = train_acc
                paddle.save(model.state_dict(), best_md_sp)
                md_logger.info(f'Best model saved at {best_md_sp}')

            # train_loss.backward()
            train_loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, val_d in enumerate(val_ties_ds):
            val_detail = get_pred_detail(val_d, model=model, config=config)
            val_loss = val_detail['row_loss']
            val_acc = val_detail['row_acc']

            accuracies.append(val_acc.numpy())
            losses.append(val_loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        md_logger.info(f"[validation] acc/loss: {val_acc.numpy()}/{val_loss.numpy()}")
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
