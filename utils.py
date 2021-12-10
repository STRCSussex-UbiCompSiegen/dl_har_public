##################################################
# Collection of various helper functions used in repository.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import errno
import os
import sys

import numpy as np
import seaborn as sn
import wandb
from matplotlib import pyplot as plt


def makedir(path):
    """
    Creates a directory if not already exists.

    :param str path: The path which is to be created.
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.exists:
        print(f"[+] Created directory in {path}")


def paint(text, color="green"):
    """
    :param text: string to be formatted
    :param color: color used for formatting the string
    :return:
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    if color == "blue":
        return OKBLUE + text + ENDC
    elif color == "green":
        return OKGREEN + text + ENDC


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":4f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            makedir(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def plot_pie(target, prefix, path_save, class_map=None, verbose=False):
    """
    Generate a pie chart of activity class distributions

    :param target: a list of activity labels corresponding to activity data segments
    :param prefix: data split, can be train, val or test
    :param path_save: path for saving the activity distribution pie chart
    :param class_map: a list of activity class names
    :param verbose:
    :return:
    """

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if not class_map:
        class_map = [str(idx) for idx in range(len(set(target)))]

    color_map = sn.color_palette(
        "husl", n_colors=len(class_map)
    )  # a list of RGB tuples

    target_dict = {
        label: np.sum(target == label_idx) for label_idx, label in enumerate(class_map)
    }
    target_count = list(target_dict.values())
    if verbose:
        print(f"[-] {prefix} target distribution: {target_dict}")
        print("--" * 50)

    fig, ax = plt.subplots()
    ax.axis("equal")
    explode = tuple(np.ones(len(class_map)) * 0.05)
    patches, texts, autotexts = ax.pie(
        target_count,
        explode=explode,
        labels=class_map,
        autopct="%1.1f%%",
        shadow=False,
        startangle=0,
        colors=color_map,
        wedgeprops={"linewidth": 1, "edgecolor": "k"},
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.set_title(dataset)
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    # plt.show()
    save_name = os.path.join(path_save, prefix + ".png")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_segment(data, target, index, prefix, path_save, num_class, target_pred=None, class_map=None):
    """
    Plot a data segment with corresonding activity label

    :param data: data segment
    :param target: ground-truth activity label corresponding to data segment
    :param index: index of segment in dataset
    :param prefix: data split, can be train, val or test
    :param path_save: path for saving the generated plot
    :param num_class: number of activity classes
    :param target_pred: predicted activity label corresponding to data segment
    :param class_map: a list of activity class names
    :return:
    """

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if not class_map:
        class_map = [str(idx) for idx in range(num_class)]

    gt = int(target)
    title_color = "black"

    if target_pred is not None:
        pred = int(target_pred)
        msg = f"#{int(index)}     ground-truth:{class_map[gt]}     prediction:{class_map[pred]}"
        title_color = "green" if gt == pred else "red"
    else:
        msg = "#{int(index)}     ground-truth:{CLASS_MAP[gt]}            "

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(data.numpy())
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(-5, 5)
    ax.set_title(msg, color=title_color)
    plt.tight_layout()
    save_name = os.path.join(
        path_save,
        prefix + "_" + class_map[int(target)] + "_" + str(int(index)) + ".png",
    )
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def wandb_logging(train_results, test_results, project, entity, config):
    wandb.init(project=project, entity=entity, config=config)

    t_loss, t_acc, t_fw, t_fm = \
        np.zeros(config['epochs']), np.zeros(config['epochs']), np.zeros(config['epochs']), np.zeros(config['epochs'])
    v_loss, v_acc, v_fw, v_fm = \
        np.zeros(config['epochs']), np.zeros(config['epochs']), np.zeros(config['epochs']), np.zeros(config['epochs'])

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
                              zip(list(range(config['epochs'])), t_loss / len(train_results), t_acc / len(train_results),
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


