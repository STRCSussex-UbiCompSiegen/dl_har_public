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

from dl_for_har_analysis.analysis import run_train_analysis, run_test_analysis

from dl_har_model.train import split_validate, loso_cross_validate
from utils import Logger, wandb_logging
from importlib import import_module

SEEDS = [1, 2]
WANDB_PROJECT = 'grokking_for_har'
WANDB_ENTITY = 'siegen-sussex-dl-for-har'

N_CLASSES = {'opportunity': 18,
             'pamap2': 12,
             'skoda': 11,
             'hhar': 0,
             'rwhar': 0}
N_CHANNELS = {'opportunity': 113,
             'pamap2': 52,
             'skoda': 60,
             'hhar': 0,
             'rwhar': 0}


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate an HAR model on given dataset.')

    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset. Required.', required=True)
    parser.add_argument(
        '-v', '--valid_type', type=str, help='Validation type. Default split.', default='split', required=False)
    parser.add_argument(
        '-m', '--model', type=str, help='Model architecture. Must be the exact name of a model in the models directory.'
                                        'Default DeepConvLSTM.', default='DeepConvLSTM')
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
        '-pf', '--print_freq', type=int, help='Print frequency (batches). Default 256.',
        default=100, required=False)
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

module = import_module(f'dl_har_model.models.{args.model}')
Model = getattr(module, args.model)

config_dataset = {
    "dataset": args.dataset,
    "window": args.window_size,
    "stride": args.window_step_train,
    "stride_test": args.window_step_test,
    "path_processed": f"data/{args.dataset}",
}

train_args = {
    "batch_size_train": args.batch_size_train,
    "batch_size_test": args.batch_size_test,
    "optimizer": args.optimizer,
    "use_weights": args.unweighted,
    "lr": args.learning_rate,
    "lr_schedule": args.learning_rate_schedule,
    "lr_step": args.learning_rate_schedule_step,
    "lr_decay": args.weight_decay,
    "weights_init": args.weights_init,
    "epochs": args.n_epochs,
    "print_freq": args.print_freq,
    "loss": args.loss,
    "smoothing": args.smoothing,
    "weight_decay": args.weight_decay
}

config = dict(
    seeds=SEEDS,
    model=args.model,
    valid_type=args.valid_type,
    batch_size_train=args.batch_size_train,
    epochs=args.n_epochs,
    optimizer=args.optimizer,
    loss=args.loss,
    smoothing=args.smoothing,
    use_weights=args.unweighted,
    lr=args.learning_rate,
    lr_schedule=args.learning_rate_schedule,
    lr_step=args.learning_rate_schedule_step,
    lr_decay=args.learning_rate_schedule_decay,
    weights_init=args.weights_init,
    weight_decay=args.weight_decay,
    batch_size_test=args.batch_size_test,
    wandb_logging=args.wandb
)

# parameters used to calculate runtime
log_date = time.strftime('%Y%m%d')
log_timestamp = time.strftime('%H%M%S')

# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log'))

model = Model(N_CHANNELS[args.dataset], N_CLASSES[args.dataset], args.dataset).cuda()

model.path_checkpoints = os.path.join('logs')
print(model)

if args.valid_type == 'split':
    train_results, test_results, preds = split_validate(model, train_args, config_dataset, SEEDS, verbose=True)
elif args.valid_type == 'loso':
    train_results, test_results, preds = loso_cross_validate(model, train_args, config_dataset, SEEDS, verbose=True)

run_train_analysis(train_results)
run_test_analysis(test_results)

if args.wandb:
    wandb_logging(train_results, test_results, WANDB_PROJECT, WANDB_ENTITY, {**config_dataset, **train_args})

if args.save_results:
    train_results.to_csv(os.path.join('logs', log_date, log_timestamp, 'train_results.csv'), index=False)
    if test_results is not None:
        test_results.to_csv(os.path.join('logs', log_date, log_timestamp, 'test_results.csv'), index=False)
    preds.to_csv(os.path.join('logs', log_date, log_timestamp, 'preds.csv'), index=False)
