##################################################
# Main script in order to execute HAR experiments
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import argparse

import wandb

from dl_har_dataloader.datasets import SensorDataset
from dl_har_model.train import cross_validate


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate an HAR model on given dataset.')

    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset. Required.', required=True)
    parser.add_argument(
        '-m', '--model', type=str, help='Model architecture. Default deepconvlstm.', default='deepconvlstm')
    parser.add_argument(
        '-e', '--n_epochs', type=int, help='Number of epochs to train. Default 300.', default=300, required=False)
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
        '--wandb', action='store_true', help='Flag indicating to log results to wandb.',
        default=False, required=False)

    args = parser.parse_args()

    return args


args = get_args()

# general settings
wandb_logging = args.wandb

# dataset settings
target_dataset = args.dataset
valid_type = 'loso'
window_size = 24
window_step = 12

# training settings
model = args.model
batch_size_train = 256
optimizer = 'Adam'
use_weights = True
lr = args.learning_rate
lr_schedule = args.learning_rate_schedule
lr_step = args.learning_rate_schedule_step
lr_decay = 0.9
weights_init = 'orthogonal'
epochs = args.n_epochs
print_freq = 100

# testing settings
batch_size_test = 256
num_batches_eval = 212
stride_test = 1


config = dict(
    dataset=target_dataset,
    model=model,
    valid_type=valid_type,
    window=window_size,
    stride=window_step,
    batch_size_train=batch_size_train,
    epochs=epochs,
    optimizer=optimizer,
    use_weights=use_weights,
    lr=lr,
    lr_schedule=lr_schedule,
    lr_step=lr_step,
    lr_decay=lr_decay,
    weights_init=weights_init,
    print_freq=print_freq,
    batch_size_test=batch_size_test,
    num_batches_eval=num_batches_eval,
    stride_test=stride_test
)

if wandb_logging:
    wandb.init(project="grokking_for_har", entity="siegen-sussex-dl-for-har", config=config)

config_dataset = {
        "dataset": config['dataset'],
        "window": config['window'],
        "stride": config['stride'],
        "stride_test": config['stride_test'],
        "path_processed": f"data/{target_dataset}",
        }

dataset = SensorDataset(**config_dataset)

config['n_classes'] = dataset.n_classes
config['n_channels'] = dataset.n_channels
config['wandb_logging'] = wandb_logging

cross_validate(dataset, valid_type, config, verbose=True)
