"""
author: ErnestinaQiu
"""
import os
import gc
import yaml
import pickle
import logging
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance
from exp.data.ties_dataset import MLDataSet
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


def exp():
    config_path = os.path.join(os.getcwd(), 'exp/configs/xgb.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    lr = config['learning_rate']
    n_estimators = config['n_estimators']

    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")
    save_dir = f'./output/exp/XGB/{date_string}'
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    ds_logger = get_logger(name='exp_ds', log_file=os.path.join(save_dir, 'exp_ds.log'), log_level=logging.CRITICAL)
    train_ds = MLDataSet(config=config, logger=ds_logger, mode='train')
    train_gen = train_ds.__iter__()
    test_ds = MLDataSet(config=config, logger=ds_logger, mode='val')
    test_gen = test_ds.__iter__()
    val_ds = MLDataSet(config=config, logger=ds_logger, mode='val')
    val_gen = val_ds.__iter__()

    md_logger = get_logger(name='md_ds', log_file=os.path.join(save_dir, 'exp_md.log'), log_level=logging.DEBUG)

    xgboost.set_config(verbosity=1, nthread=4)

    es = xgboost.callback.EarlyStopping(
        rounds=30,
        min_delta=1e-4,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name="mlogloss",
    )

    model = XGBClassifier(max_depth=10,
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
                        subsample=0.85,
                        colsample_bytree=0.9,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=123,
                        missing=None,
                        eval_metric='auc',
                        callbacks=[es],
                        )

    train_df = next(train_gen)
    train_sp = os.path.join(data_dir, 'train.csv')
    train_df.to_csv(train_sp, index=False, header=True, encoding='utf8')
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    train_set_desc = train_df['label'].value_counts(normalize=True)

    test_df = next(test_gen)
    test_sp = os.path.join(data_dir, 'test.csv')
    test_df.to_csv(test_sp, header=True, encoding='utf8')
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    test_set_desc = test_df['label'].value_counts(normalize=True)

    val_df = next(val_gen)
    val_sp = os.path.join(data_dir, 'test.csv')
    val_df.to_csv(val_sp, header=True, encoding='utf8')
    x_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, :-1]
    val_set_desc = val_df['label'].value_counts(normalize=True)
    md_logger.info(f"train set distribution:\n{train_set_desc}\ntest set distribution:\n{test_set_desc}\nval set distribution:\n{val_set_desc}")

    model.fit(x_train, y_train, verbose=10, eval_set=[(x_val, y_val)])

    y_pred = model.predict(x_test, ntree_limit=model.best_ntree_limit)
    auc_score = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    confusion_matrix_result = metrics.confusion_matrix(y_pred, y_test)

    md_logger.info(f"[test] auc_score: {auc_score} acc: {acc}\nThe confusion matrix result:\n{confusion_matrix_resul}\nfeature importance:\n{model.feature_importances_}")

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
    exp()