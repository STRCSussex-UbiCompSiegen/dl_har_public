from dl_har_model.models.DeepConvLSTM import DeepConvLSTM
from dl_har_model.models.AttendAndDiscriminate import AttendDiscriminate
from dl_har_dataloader.datasets import SensorDataset
from dl_har_model.train import loso_cross_validate, train_model
from dl_har_model.eval import eval_model

model = DeepConvLSTM(113, 18, 'opportunity').cuda()

config_dataset = {
    "dataset": 'opportunity',
    "window": 24,
    "stride": 12,
    "stride_test": 1,
    "path_processed": f"./data/opportunity",
}

train_args = {'epochs': 1}
num_users = 4

loso_cross_validate(model, num_users, train_args, config_dataset, verbose=True)

train_data = SensorDataset(prefix='User_0', **config_dataset)
val_data = SensorDataset(prefix='User_1', **config_dataset)

t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, _ = train_model(model, train_data, val_data, verbose=True)

attend_discriminate_config = dict(n_channels=113,
                                  dataset='opportunity',
                                  activation='Relu',
                                  dropout=0.5,
                                  dropout_cls=0.5,
                                  enc_is_bidirectional=False,
                                  enc_num_layers=2,
                                  experiment='train',
                                  filter_num=64,
                                  filter_size=5,
                                  hidden_dim=128,
                                  num_class=18,
                                  sa_div=1,
                                  train_mode=True)

model = AttendDiscriminate(**attend_discriminate_config).cuda()

loso_cross_validate(model, num_users, train_args, config_dataset, verbose=True)

train_data = SensorDataset(prefix='User_0', **config_dataset)
val_data = SensorDataset(prefix='User_1', **config_dataset)

t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, _ = train_model(model, train_data, val_data, verbose=True,
                                                                   **train_args)

test_data = SensorDataset(prefix='User_2', **config_dataset)
loss_test, acc_test, fm_test, fw_test, elapsed, preds = eval_model(model, test_data)
