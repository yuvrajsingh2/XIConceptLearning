import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import sys
sys.path.append("Proto_Cat_VAE/models/")
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

from torch.optim import Adam

import VQ_VAE.utils_proto as utils
import VQ_VAE.data as data
from VQ_VAE.models.gproto_ae_sup import GProtoAE
from VQ_VAE.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    cls_criterion = nn.CrossEntropyLoss()
    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'recon_loss_z': 0, 'recon_loss_proto': 0, 'loss': 0, 'vq_loss': 0, 'cls_loss': 0})

        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(data_loader):
            imgs = batch[0]
            labels = batch[1]
            # reshape to have prediction for each attribute
            labels = labels.reshape(-1, config['n_groups'], config['n_protos']).permute(1, 0, 2)
            # TODO: hack because we don'' have custom dataloader, we convert one hot to class id
            labels = torch.argmax(labels, dim=2) # [B, G]

            imgs = imgs.to(config['device'])
            labels = labels.to(config['device'])

            std = (config['epochs'] - e) / config['epochs']

            vq_loss, imgs_recon_z, imgs_recon_proto, _, _, _, attr_preds = model(imgs, labels)

            # reconstruciton loss for z and protos
            recon_loss_z = F.mse_loss(imgs_recon_z, imgs)
            recon_loss_proto = F.mse_loss(imgs_recon_proto, imgs)

            # compute CE loss for each attribute and average over groups
            cls_loss = torch.mean(torch.stack([cls_criterion(attr_preds[i], labels[i])
                                               for i in range(config['n_groups'])]))
            loss = config['lambda_recon_z'] * recon_loss_z + \
                   config['lambda_recon_proto'] * recon_loss_proto + \
                   vq_loss + \
                   config['lambda_cls'] * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['recon_loss_z'] += recon_loss_z.item() if config['lambda_recon_z'] > 0. else 0.
            loss_dict['recon_loss_proto'] += recon_loss_proto.item() if config['lambda_recon_proto'] > 0. else 0.
            loss_dict['cls_loss'] += config['lambda_cls'] * cls_loss.item() if config['lambda_cls'] > 0. else 0.
            loss_dict['vq_loss'] += vq_loss.item()
            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e + 1) % config['display_step'] == 0 or e == config['epochs'] - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=e)
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

        if (e + 1) % config['print_step'] == 0 or e == config['epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e + 1) % config['save_step'] == 0 or e == config['epochs'] - 1 or e == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ep': e,
                'config': config
            }
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            # plot the individual prototypes of each group
            utils.plot_prototypes(model, writer, config, step=e)

            # plot a few samples with proto recon
            utils.plot_examples(log_samples, model, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def main(config):

    # get train data
    _data_loader = data.get_dataloader(config)

    # get test set samples
    test_set = data.get_test_set(_data_loader, config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # TODO fix add_hparams
    # list_key = []
    # for key in config.keys():
    #     if type(config[key]) is type(list()):
    #         list_key += [key]

    # for key in list_key:
    #     for i, item in enumerate(config[key]):
    #         config[key+str(i)] = item
    #     del config[key]

    # writer.add_hparams(config, dict())

    # model setup
    _model = GProtoVAE(num_hiddens=32, num_residual_layers=2, num_residual_hiddens=32,
                   num_groups=config['n_groups'], num_protos=config['n_protos'],
                   commitment_cost=config['lambda_commitment_cost'], agg_type=config['agg_type'],
                   device=config['device'])

    _model = _model.to(config['device'])

    # optimizer setup
    optimizer = torch.optim.Adam(_model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    scheduler = None
    if config['lr_scheduler']:
        # TODO: try LambdaLR
        num_steps = len(_data_loader) * config['epochs']
        num_steps += config['lr_scheduler_warmup_steps']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    # start training
    train(_model, _data_loader, test_set, optimizer, scheduler, writer, config)


if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)