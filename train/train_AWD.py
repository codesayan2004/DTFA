import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.AWD.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from nets.AWD.shallow_darknet import ShallowNet
from utils.AFB.loss import AFBLoss
from utils.DBT.loss import DBTLoss
from utils.callbacks import LossHistory
from utils.AWD_dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.fit import fit_one_epoch

if __name__ == "__main__":
    # Settings can be overridden with environment variables (see COLAB.md).
    Cuda = torch.cuda.is_available()
    classes_path = 'model_data/classes.txt'
    model_path = os.environ.get('MODEL_PATH', 'model_data/yolox_s.pth')
    SPT_model_path = os.environ.get('SPT_PATH', 'model_data/SPT.pth')
    IRT_model_path = os.environ.get('IRT_PATH', 'model_data/IRT.pth')

    input_shape = [640, 640]
    phi = 's'
    mosaic = False

    Init_Epoch = int(os.environ.get('INIT_EPOCH', 0))
    Freeze_Epoch = 0
    Freeze_batch_size = 8

    UnFreeze_Epoch = int(os.environ.get('EPOCHS', 100))
    Unfreeze_batch_size = int(os.environ.get('BATCH_SIZE', 8))

    Freeze_Train = False

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4

    lr_decay_type = "cos"

    save_period = int(os.environ.get('SAVE_PERIOD', 50))

    num_workers = int(os.environ.get('NUM_WORKERS', 4))

    train_annotation_path = os.environ.get('TRAIN_ANN', 'datasets/data_info/train_fog.txt')
    val_annotation_path = os.environ.get('VAL_ANN', 'datasets/data_info/val_fog.txt')
    clear_annotation_path = 'datasets/data_info/train_clear.txt'
    val_clear_annotation_path = 'datasets/data_info/val_clear.txt'

    class_names, num_classes = get_classes(classes_path)

    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
        #for k, v in pretrained_dict.items():
            #print(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    yolo_loss = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/", model, input_shape=input_shape)
    model_train = model.train()
    if Cuda:
        cudnn.benchmark = True
        model_train = model_train.cuda()


    'IRT'
    res_model = ShallowNet()
    weights_init(res_model)
    print('Load restoration branch weights {}.'.format(IRT_model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_model_dict = res_model.state_dict()
    res_pretrained_dict = torch.load(IRT_model_path, map_location=device)
    res_pretrained_dict_filtered = {}
    for k, v in res_pretrained_dict.items():
        k = k.replace('backbone.backbone.', 'backbone.', 1)
        if k in res_model_dict and res_model_dict[k].shape == v.shape:  # 注意使用.shape而不是np.shape()
            res_pretrained_dict_filtered[k] = v
    res_model_dict.update(res_pretrained_dict_filtered)
    print('  matched {}/{} teacher tensors'.format(len(res_pretrained_dict_filtered), len(res_model_dict)))
    if len(res_pretrained_dict_filtered) < len(res_model_dict):
        print('  WARNING: teacher checkpoint only partially matches ShallowNet')
    res_model.load_state_dict(res_model_dict)
    res_model = res_model.eval()
    for (name, param) in res_model.named_parameters():
        param.requires_grad = False
        #print(name)
    if Cuda:
        res_model = res_model.cuda()

    'SPT'
    det_model = ShallowNet()
    weights_init(det_model)
    print('Load detection branch weights {}.'.format(SPT_model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_model_dict = det_model.state_dict()
    det_pretrained_dict = torch.load(SPT_model_path, map_location=device)
    det_pretrained_dict_filtered = {}
    for k, v in det_pretrained_dict.items():
        k = k.replace('backbone.backbone.', 'backbone.', 1)
        if k in det_model_dict and det_model_dict[k].shape == v.shape:  # 注意使用.shape而不是np.shape()
            det_pretrained_dict_filtered[k] = v
    det_model_dict.update(det_pretrained_dict_filtered)
    print('  matched {}/{} teacher tensors'.format(len(det_pretrained_dict_filtered), len(det_model_dict)))
    if len(det_pretrained_dict_filtered) < len(det_model_dict):
        print('  WARNING: teacher checkpoint only partially matches ShallowNet')
    det_model.load_state_dict(det_model_dict)
    det_model = det_model.eval()
    for (name, param) in det_model.named_parameters():
        param.requires_grad = False
        #print(name)
    if Cuda:
        det_model = det_model.cuda()

    # Dynamic Bridging Tuning (utils/DBT/loss.py). DBT=0 (default) runs the original DTFA.
    use_dbt = os.environ.get('DBT', '0') == '1'
    dbt = None
    D_guidance = R_guidance = None
    if use_dbt:
        flag = lambda name, default='1': os.environ.get(name, default) == '1'
        dbt = DBTLoss(alpha_mgd=0.0000001,
                      mode=os.environ.get('DBT_MODE', 'spatial'),
                      use_student=flag('DBT_USE_S'), use_severity=flag('DBT_USE_W'),
                      use_disagreement=flag('DBT_USE_D'),
                      lam=float(os.environ.get('DBT_LAMBDA', 2.0)),
                      eta=float(os.environ.get('DBT_ETA', 1.0)),
                      gamma=float(os.environ.get('DBT_GAMMA', 0.01)),
                      mu=float(os.environ.get('DBT_MU', 0.1)),
                      sev_scale=float(os.environ.get('DBT_SEV_SCALE', 0.25)))
        weights_init(dbt)   # same N(0, 0.02) init as the baseline AFB bridges
        dbt.reset_gate()    # gate starts from an even IRT/SPT split
        dbt_path = os.environ.get('DBT_PATH', '')
        if not dbt_path and Init_Epoch > 0:  # resuming: look for the matching gate checkpoint
            for d in (os.path.dirname(model_path), 'logs'):
                cand = os.path.join(d, 'dbt_ep%03d.pth' % Init_Epoch)
                if os.path.exists(cand):
                    dbt_path = cand
                    break
        # The gate gets its own Adam optimizer: with the student's SGD learning rate (~1e-3) the gate
        # barely moves (checked on a synthetic test), with Adam it learns the per-region teacher choice.
        dbt.optimizer = optim.Adam(dbt.trainable_parameters(), float(os.environ.get('DBT_LR', 1e-3)))
        if dbt_path:
            ckpt = torch.load(dbt_path, map_location='cpu')
            dbt.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
            if 'optimizer' in ckpt:
                dbt.optimizer.load_state_dict(ckpt['optimizer'])
            print('Load DBT weights {}.'.format(dbt_path))
        elif Init_Epoch > 0:
            print('WARNING: resuming a DBT run without dbt_ep%03d.pth; the gate restarts from scratch' % Init_Epoch)
        if Cuda:
            dbt = dbt.cuda()
            for st in dbt.optimizer.state.values():  # optimizer state restored on CPU
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.cuda()
        print('Dynamic Bridging Tuning: ' + dbt.describe() + ', gate optimizer=Adam(lr={})'.format(
            dbt.optimizer.param_groups[0]['lr']))
    else:
        'D_guidance'
        D_guidance = AFBLoss(alpha_mgd=0.0000001)
        weights_init(D_guidance)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        D_guidance = D_guidance.train()
        if Cuda:
            D_guidance = D_guidance.cuda()
        print('Detection Guidance Down')
        'R_guidance'
        R_guidance = AFBLoss(alpha_mgd=0.0000001)
        weights_init(R_guidance)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R_guidance = R_guidance.train()
        if Cuda:
            R_guidance = R_guidance.cuda()


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    with open(clear_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()
    with open(val_clear_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()
    # Optional smaller run: TRAIN_SUBSET / VAL_SUBSET = number of images to use. The same random
    # indices are taken from the fog and clean lists so the pairs stay aligned.
    import random
    for env, fog, clr in (('TRAIN_SUBSET', train_lines, clear_lines), ('VAL_SUBSET', val_lines, val_clear_lines)):
        n = int(os.environ.get(env, 0))
        if 0 < n < len(fog):
            idx = sorted(random.Random(0).sample(range(len(fog)), n))
            fog[:] = [fog[i] for i in idx]
            clr[:] = [clr[i] for i in idx]
    num_train = len(train_lines)
    num_val = len(val_lines)
    print('Train images: {}, val images: {}'.format(num_train, num_val))

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset error!")

        train_dataset = YoloDataset(train_lines, clear_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, val_clear_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                  mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # STOP_EPOCH ends this run early (the lr schedule still spans EPOCHS); resume later with INIT_EPOCH.
        Stop_Epoch = min(UnFreeze_Epoch, int(os.environ.get('STOP_EPOCH', UnFreeze_Epoch)))
        for epoch in range(Init_Epoch, Stop_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset error！")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, save_period, res_model, det_model,R_guidance , D_guidance,
                          dbt=dbt)