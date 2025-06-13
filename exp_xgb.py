"""
author: ErnestinaQiu
"""
import os
import gc
import yaml
import shutil
import pickle
import logging
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance
from exp.data.ties_dataset import MLDataSet
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


def get_val_data():
    config_path = os.path.join(os.getcwd(), 'exp/configs/xgb.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data_dir = './output/exp/XGB/data'
    os.makedirs(data_dir, exist_ok=True)

    ds_logger = get_logger(name='exp_ds', log_file=os.path.join(data_dir, 'exp_ds.log'), log_level=logging.CRITICAL)

    ds = MLDataSet(config=config, logger=ds_logger, mode='val')
    save_path = os.path.join(data_dir, 'all_val_ds.csv')
    ds.GetDataSet(save_path=save_path)


def gen_data():
    config_path = os.path.join(os.getcwd(), 'exp/configs/xgb.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data_dir = './output/exp/XGB/data'
    os.makedirs(data_dir, exist_ok=True)

    ds_logger = get_logger(name='exp_ds', log_file=os.path.join(data_dir, 'exp_ds.log'), log_level=logging.CRITICAL)
    train_ds = MLDataSet(config=config, logger=ds_logger, mode='train')
    train_gen = train_ds.__iter__()
    test_ds = MLDataSet(config=config, logger=ds_logger, mode='val')
    test_gen = test_ds.__iter__()
    val_ds = MLDataSet(config=config, logger=ds_logger, mode='val')
    val_gen = val_ds.__iter__()

    train_df = next(train_gen)
    train_sp = os.path.join(data_dir, 'train.csv')
    i = 1
    while os.path.exists(train_sp):
        train_sp = os.path.join(data_dir, f'train_{i}.csv')
        i += 1

    train_df.to_csv(train_sp, index=False, header=True, encoding='utf8')
    train_set_desc = train_df['label'].value_counts(normalize=True)

    test_df = next(test_gen)
    test_sp = os.path.join(data_dir, 'test.csv')
    i = 1
    while os.path.exists(test_sp):
        test_sp = os.path.join(data_dir, f'test_{i}.csv')
        i += 1

    test_df.to_csv(test_sp, index=False, header=True, encoding='utf8')
    test_set_desc = test_df['label'].value_counts(normalize=True)

    val_df = next(val_gen)
    val_sp = os.path.join(data_dir, 'val.csv')
    i = 1
    while os.path.exists(val_sp):
        val_sp = os.path.join(data_dir, f'val_{i}.csv')
        i += 1

    val_df.to_csv(val_sp, index=False, header=True, encoding='utf8')
    val_set_desc = val_df['label'].value_counts(normalize=True)

    ds_logger.critical(f"train set distribution:\n{train_set_desc}\ntest set distribution:\n{test_set_desc}\nval set distribution:\n{val_set_desc}")


def test():
    save_dir = './output/exp/XGB/test'
    save_dir = os.path.join(save_dir, 'test_all_val')
    os.makedirs(save_dir, exist_ok=True)

    md_logger = get_logger(name='md_ds', log_file=os.path.join(save_dir, 'exp_md.log'), log_level=logging.DEBUG)

    md_path = './output/exp/XGB/20250610115334/xgb.pickle'
    model = pickle.load(open(md_path, 'rb'))

    test_ds_path = './output/exp/XGB/data/all_val_ds.csv'
    test_df = pd.read_csv(test_ds_path)
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    test_set_desc = test_df['label'].value_counts(normalize=True)

    md_logger.debug(f'test_set_desc: {test_set_desc}')

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    confusion_matrix_result = metrics.confusion_matrix(y_pred, y_test)

    cell_acc = confusion_matrix_result[0, 0] / np.sum(confusion_matrix_result[0, :])
    cell_precision = confusion_matrix_result[0, 0] / np.sum(confusion_matrix_result[:, 0])
    row_acc = confusion_matrix_result[1, 1] / np.sum(confusion_matrix_result[1, :])
    row_precision = confusion_matrix_result[1, 1] / np.sum(confusion_matrix_result[1, :])
    col_acc = confusion_matrix_result[2, 2] / np.sum(confusion_matrix_result[2, :])
    col_precision = confusion_matrix_result[2, 2] / np.sum(confusion_matrix_result[:, 2])
    no_rel_acc = confusion_matrix_result[3, 3] / np.sum(confusion_matrix_result[3, :])
    no_rel_precision = confusion_matrix_result[3, 3] / np.sum(confusion_matrix_result[:, 3])
    cat_acc = {'cell_acc': cell_acc, 'cell_precision': cell_precision, 'row_acc': row_acc, 'row_precision': row_precision, 'col_acc': col_acc, 'col_precision': col_precision, 'no_rel_acc': no_rel_acc, 'no_rel_precision': no_rel_precision}

    md_logger.info(f"[test] acc: {acc}\ncat_acc: {cat_acc}\nfeature importance:\n{list(model.feature_importances_)}\nmodel params: {model.get_params()}")

    confusion_matrix_sp = os.path.join(save_dir, 'confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(confusion_matrix_sp)
    plt.close()

    md_logger.info(f'Output result into {save_dir}')


def exp():
    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")
    save_dir = f'./output/exp/XGB/{date_string}'

    md_logger = get_logger(name='md_ds', log_file=os.path.join(save_dir, 'exp_md.log'), log_level=logging.DEBUG)

    data_dir = './output/exp/XGB/data'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    val_path = os.path.join(data_dir, 'val.csv')

    train_df = pd.read_csv(train_path)
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    train_set_desc = train_df['label'].value_counts(normalize=True)

    test_df = pd.read_csv(test_path)
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    test_set_desc = test_df['label'].value_counts(normalize=True)

    val_df = pd.read_csv(val_path)
    x_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]
    val_set_desc = val_df['label'].value_counts(normalize=True)

    md_logger.info(f"train set distribution:\n{train_set_desc}\ntest set distribution:\n{test_set_desc}\nval set distribution:\n{val_set_desc}")

    config_path = os.path.join(os.getcwd(), 'exp/configs/xgb.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    lr = config['learning_rate']
    n_estimators = config['n_estimators']

    xgboost.set_config(verbosity=1, nthread=4)

    es = xgboost.callback.EarlyStopping(
        rounds=50,
        min_delta=1e-4,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name="merror",
    )

    model = XGBClassifier(max_depth=5,
                        device="cuda",
                        learning_rate=lr,
                        n_estimators=n_estimators,
                        silent=True,
                        num_class=4,
                        objective='multi:softmax',
                        nthread=4,
                        gamma=0,
                        min_child_weight=5,
                        max_delta_step=0,
                        subsample=1,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=123,
                        missing= -999999,
                        eval_metric='merror',
                        callbacks=[es],
                        )

    model.fit(x_train, y_train, verbose=10, eval_set=[(x_val, y_val)])

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    confusion_matrix_result = metrics.confusion_matrix(y_pred, y_test)

    md_logger.info(f"[test] acc: {acc}\nfeature importance:\n{list(model.feature_importances_)}\nmodel params: {model.get_params()}\nconfig:\n{config}")

    confusion_matrix_sp = os.path.join(save_dir, 'confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(confusion_matrix_sp)
    plt.close()

    model_sp = os.path.join(save_dir, 'xgb.json')
    model.save_model(model_sp)

    pk_model_sp = os.path.join(save_dir, 'xgb.pickle')
    pickle.dump(model, open(pk_model_sp, 'wb'))

    md_logger.info(f'Save model into {model_sp}')


if __name__ == "__main__":
    # get_val_data()
    test()

    # gen_data()
    # exp()