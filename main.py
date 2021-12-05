##################################################
# Main script in order to execute HAR experiments
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import argparse
import os
import sys
import time

import numpy as np
import wandb

from dl_har_model.train import cross_validate
from utils import Logger, paint

SEEDS = [1, 2, 3, 4, 5]
WANDB_PROJECT = 'grokking_for_har'
WANDB_ENTITY = 'siegen-sussex-dl-for-har'


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate an HAR model on given dataset.')

    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset. Required.', required=True)
    parser.add_argument(
        '-v', '--valid_type', type=str, help='Validation type. Default split.', default='split', required=False)
    parser.add_argument(
        '-m', '--model', type=str, help='Model architecture. Default deepconvlstm.', default='deepconvlstm')
    parser.add_argument(
        '-e', '--n_epochs', type=int, help='Number of epochs to train. Default 300.', default=300, required=False)
    parser.add_argument(
        '-o', '--optimizer', type=str, help='Optimizer. Default adam.', default='adam',
        required=False)
    parser.add_argument(
        '-l', '--loss', type=str, help='Loss calculation. Default cross-entropy.', default='cross-entropy',
        required=False)
    parser.add_argument(
        '-s', '--smoothing', type=float, help='Label smoothing. Default 0.0.', default=0.0, required=False)
    parser.add_argument(
        '-w', '--weights_init', type=str, help='Weight initialization. Default orthogonal.', default='orthogonal',
        required=False)
    parser.add_argument(
        '-wd', '--weight_decay', type=float, help='Weight decay. Default 0.0.', default=0.0,
        required=False)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, help='Initial learning rate. Default 1e-3.', default=1e-3, required=False)
    parser.add_argument(
        '-ls', '--learning_rate_schedule', type=str, help='Type of learning rate schedule. Default step.',
        default='step', required=False)
    parser.add_argument(
        '-lss', '--learning_rate_schedule_step', type=int,
        help='Initial learning rate schedule step size. If 0, learning rate schedule not applied. Default 10.',
        default=10, required=False)
    parser.add_argument(
        '-lsd', '--learning_rate_schedule_decay', type=float, help='Learning rate schedule decay. Default 0.9.',
        default=0.9, required=False)
    parser.add_argument(
        '-ws', '--window_size', type=int, help='Sliding window size. Default 24.',
        default=24, required=False)
    parser.add_argument(
        '-wstr', '--window_step_train', type=int, help='Sliding window step size train. Default 12.',
        default=12, required=False)
    parser.add_argument(
        '-wste', '--window_step_test', type=int, help='Sliding window step size test. Default 1.',
        default=1, required=False)
    parser.add_argument(
        '-bstr', '--batch_size_train', type=int, help='Batch size training. Default 256.',
        default=256, required=False)
    parser.add_argument(
        '-bste', '--batch_size_test', type=int, help='Batch size testing. Default 256.',
        default=256, required=False)
    parser.add_argument(
        '--wandb', action='store_true', help='Flag indicating to log results to wandb.',
        default=False, required=False)
    parser.add_argument(
        '--logging', action='store_true', help='Flag indicating to log results locally.',
        default=False, required=False)
    parser.add_argument(
        '--save_results', action='store_true', help='Flag indicating to save results.',
        default=False, required=False)
    parser.add_argument(
        '--unweighted', action='store_false', help='Flag indicating to use unweighted loss.',
        default=True, required=False)

    args = parser.parse_args()

    return args


args = get_args()

# general settings
wandb_logging = args.wandb
log_results = args.logging
save_results = args.save_results

# dataset settings
target_dataset = args.dataset
valid_type = args.valid_type
window_size = args.window_size
window_step_train = args.window_step_train
window_step_test = args.window_step_test


# training settings
model = args.model
batch_size_train = args.batch_size_train
optimizer = args.optimizer
loss = args.loss
smoothing = args.smoothing
use_weights = args.unweighted
lr = args.learning_rate
lr_schedule = args.learning_rate_schedule
lr_step = args.learning_rate_schedule_step
lr_decay = args.learning_rate_schedule_decay
weights_init = args.weights_init
weight_decay = args.weight_decay
epochs = args.n_epochs

# testing settings
batch_size_test = args.batch_size_test

config = dict(
    seeds=SEEDS,
    dataset=target_dataset,
    model=model,
    valid_type=valid_type,
    window=window_size,
    stride=window_step_train,
    batch_size_train=batch_size_train,
    epochs=epochs,
    optimizer=optimizer,
    loss=loss,
    smoothing=smoothing,
    use_weights=use_weights,
    lr=lr,
    lr_schedule=lr_schedule,
    lr_step=lr_step,
    lr_decay=lr_decay,
    weights_init=weights_init,
    weight_decay=weight_decay,
    batch_size_test=batch_size_test,
    stride_test=window_step_test,
    wandb_logging=wandb_logging
)

# parameters used to calculate runtime
start = time.time()
log_date = time.strftime('%Y%m%d')
log_timestamp = time.strftime('%H%M%S')

# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log'))

if wandb_logging:
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config)

train_results, test_results, preds = cross_validate(valid_type, config, verbose=True)

if wandb_logging:
    t_loss, t_acc, t_fw, t_fm = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
    v_loss, v_acc, v_fw, v_fm = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)

    for i in range(len(train_results)):
        t_loss = np.add(t_loss, train_results['t_loss'][i])
        t_acc = np.add(t_acc, train_results['t_acc'][i])
        t_fw = np.add(t_fw, train_results['t_fw'][i])
        t_fm = np.add(t_fm, train_results['t_fm'][i])

        v_loss = np.add(v_loss, train_results['v_loss'][i])
        v_acc = np.add(v_acc, train_results['v_acc'][i])
        v_fw = np.add(v_fw, train_results['v_fw'][i])
        v_fm = np.add(v_fm, train_results['v_fm'][i])

    table = wandb.Table(data=[[a, b, c, d, e, f, g, h, i] for (a, b, c, d, e, f, g, h, i) in
                              zip(list(range(epochs)), t_loss / len(train_results), t_acc / len(train_results),
                                  t_fw / len(train_results), t_fm / len(train_results), v_loss / len(train_results),
                                  v_acc / len(train_results), t_fw / len(train_results), t_fm / len(train_results))],
                        columns=["epochs", "t_loss", "t_acc", "t_fw", "t_fm", "v_loss", "v_acc", "f_fw", "f_fm"]
                        )

    wandb.log({"train_loss": wandb.plot.line(table, "epochs", "t_loss", title='Train Loss'),
               "train_acc": wandb.plot.line(table, "epochs", "t_acc", title='Train Accuracy'),
               "train_fm": wandb.plot.line(table, "epochs", "t_fm", title='Train F1-macro'),
               "train_fw": wandb.plot.line(table, "epochs", "t_fw", title='Train F1-weighted'),
               "val_loss": wandb.plot.line(table, "epochs", "v_loss", title='Valid Loss'),
               "val_acc": wandb.plot.line(table, "epochs", "v_acc", title='Valid Accuracy'),
               "val_fm": wandb.plot.line(table, "epochs", "v_fm", title='Valid F1-macro'),
               "val_fw": wandb.plot.line(table, "epochs", "v_fw", title='Valid F1-weigthed')})

    if test_results is not None:
        wandb.log({"test_loss": test_results['test_loss'].mean(),
                   "test_acc": test_results['test_acc'].mean(),
                   "test_fm": test_results['test_fm'].mean(),
                   "test_fw": test_results['test_fw'].mean()
                   })

if save_results:
    train_results.to_csv(os.path.join('logs', log_date, log_timestamp, 'train_results.csv'), index=False)
    if test_results is not None:
        test_results.to_csv(os.path.join('logs', log_date, log_timestamp, 'test_results.csv'), index=False)
    preds.to_csv(os.path.join('logs', log_date, log_timestamp, 'preds.csv'), index=False)

# calculate time data creation took
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(paint("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds)))
