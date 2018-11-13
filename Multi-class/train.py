# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:37:33 2018

@author: lmx
"""
import copy
import argparse
import collections
import datetime
import os
import shutil

import mxnet as mx
import mxnet.gluon.data.vision.transforms as T
from colorama import init, Fore
from mxboard import SummaryWriter
from mxnet import gluon, nd, autograd
from mxnet.gluon.model_zoo import vision, model_store
from prettytable import PrettyTable
from tqdm import tqdm

from network import DRML, PretrainedModel, VGGFace
import math
import numpy as np
from sklearn.preprocessing import normalize
from mxnet import ndarray as nd
from mxnet.gluon.loss import Loss
from data import ImageRecordDataset
from mxnet.gluon.block import HybridBlock

init(autoreset=True)


class DataLoaderWrapper(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._gen = self.__gen__()

    def __next__(self):
        return next(self._gen)

    def __gen__(self):
        while True:
            for i in self.data_loader:
                yield i


class Transpose(gluon.HybridBlock):
    def __init__(self):
        super(Transpose, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.cast(F.transpose(x, (2, 0, 1)), dtype='float32')


def build_network(name, classes, checkpoint=None, ctx=mx.cpu(), **kwargs):
    if name == 'mobileface':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', 'model',
                                      'model-y1-test2', 'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', 'model',
                                      'model-y1-test2', 'model-0000.params')
            net = PretrainedModel(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net


def get_transforms(name):
    if name in ['r50', 'mobileface', 'dpn68', 'd121']:
        train_transform = T.Compose([T.RandomFlipLeftRight(), Transpose()])
        eval_transform = T.Compose([Transpose()])
    else:
        raise ValueError("Invalid Network Input")
    return train_transform, eval_transform


class Arcface_loss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(Arcface_loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, all_label, _weight, fc, args, sample_weight=None):
        gt_label = all_label
        _weight = _weight
        embedding = fc
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = F.L2Normalization(_weight, mode='instance')
        nembedding = F.L2Normalization(embedding, mode='instance') * s
        fc7 = F.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                               name='fc7')
        zy = F.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        # threshold = 0.0
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = F.maximum(cos_t, 0)
        else:
            cond_v = cos_t - threshold
            cond = F.maximum(cond_v, 0)
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = F.sqrt(body)
        new_zy = cos_t * cos_m
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - s * mm
        new_zy = F.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = F.expand_dims(diff, 1)
        gt_one_hot = F.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = F.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body
        # out = F.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
        if args.ce_loss:
            body = F.SoftmaxActivation(data=fc7)
            body = F.log(F.clip(body, 1e-25, 1.0))
            _label = F.one_hot(gt_label, depth=args.num_classes, on_value=-1.0, off_value=0.0)
            body = body * _label
            out = F.sum(body) / len(body)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-classes', '-num', type=int, default=22, help='classes number')
    parser.add_argument('--record', '-rec', nargs='+', default='../rec_version4/train.rec', help='path of .rec file for training')
    parser.add_argument('--eval-record', '-e', nargs='+', help='path of .rec file for evaluating')
    parser.add_argument('--output-dir', '-o', default='../model_version4/out/', help='directory for logging and saving model')
    parser.add_argument('--save-freq', '-sf', type=int, default=500, help='save model every n steps')
    parser.add_argument('--eval-freq', '-ef', type=int, default=500, help='evaluating model every n steps')
    parser.add_argument('--print-freq', '-pf', type=int, default=50, help='print logs in terminal every n steps')

    parser.add_argument('--restart', '-r', action='store_true', help='ignore any checkpoints')
    parser.add_argument('--freeze-backbone', '-fb', action='store_true', help='freeze the params in backbone')
    parser.add_argument('--freeze-steps', '-fs', type=int, default=0, help='max steps for freezing the params')
    parser.add_argument('--gpu-device', '-gpu', type=int, required=True, help='specify gpu id')

    parser.add_argument('--network', '-net', default='mobileface', help='choose network')
    parser.add_argument('--max-steps', '-ms', type=int, default=10000, help='max steps for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=256, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--optimizer', '-opt', default='adam', choices=['adam', 'sgd', 'nag'], help='type of optimizer')
    parser.add_argument('--decay-params', '-dp', nargs=2, type=float, default=[500, 0.8], help='decay step and decay rate of learning rate')
    parser.add_argument('--enable-balance-sampler', '-b', action='store_true',
                        help='enable balance sampler in batching during training')

    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--margin-m', type=float, default=0.3, help='margin for loss')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--easy-margin', type=int, default=0, help='')
    parser.add_argument('--ce-loss', default=True, action='store_true', help='if output ce loss')

    args = parser.parse_args()

    arg_table = PrettyTable(['Argument', 'Value'])
    arg_table.align['Argument'] = 'r'
    arg_table.align['Value'] = 'l'
    for k, v in vars(args).items():
        if isinstance(v, (list, tuple)) and isinstance(v[0], str):
            v = '\n'.join(v)
        arg_table.add_row([k, v])
    print(arg_table)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    args.record = [os.path.abspath(os.path.expanduser(fname)) for fname in args.record]
    args.eval_record = [os.path.abspath(os.path.expanduser(fname)) for fname in
                        args.eval_record] if args.eval_record else None
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir)) if args.output_dir else os.path.join(
        root_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    if args.restart:
        tqdm.write('Existing models and logs will be removed. [{}]'.format(args.output_dir))
        while True:
            confirm = input('Sure to restart? (y)es|(n)o: ').strip()
            if confirm in ['yes', 'y']:
                shutil.rmtree(args.output_dir, ignore_errors=True)
                tqdm.write('Directory removed: {}'.format(args.output_dir))
                break
            elif confirm in ['no', 'n']:
                break

    if args.freeze_backbone:
        print(Fore.CYAN + f'Freeze the params in backbone during first {args.freeze_steps} steps')
    
    # Create summary dirs
    train_log_dir = os.path.join(args.output_dir, 'log', 'train')
    eval_log_dir = os.path.join(args.output_dir, 'log', 'eval')
    model_dir = os.path.join(args.output_dir, 'checkpoint')
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Store parameter config
    with open(os.path.join(args.output_dir, 'train_params.txt'), 'w') as f:
        f.write(arg_table.get_string())


    ctx = mx.gpu(args.gpu_device) if args.gpu_device else mx.cpu()

    # Dataloader
    train_transform, eval_transform = get_transforms(args.network)
    record = args.record[0]
    train_dataloader = DataLoaderWrapper(gluon.data.DataLoader(
        dataset=gluon.data.vision.ImageRecordDataset(
            filename=record,
            transform=lambda x, y: (train_transform(x), y)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        last_batch='rollover'
    ))

    if args.eval_record:
        eval_record = args.eval_record[0]
        eval_dataloader = gluon.data.DataLoader(
            dataset=gluon.data.vision.ImageRecordDataset(
                filename=eval_record, transform=lambda x, y: (train_transform(x), y)
            ),
            batch_size=args.batch_size,
            shuffle=False,
            last_batch='keep'
        )

    # Restore last checkpoint
    checkpoint_files = sorted([fname for fname in os.listdir(model_dir) if fname.endswith('.params')])
    last_checkpoint = os.path.join(model_dir, checkpoint_files[-1]) if checkpoint_files else None
    start_step = int(os.path.splitext(last_checkpoint)[0][-4:]) * 100 + 1 if last_checkpoint else 1

    net = build_network(args.network, args.num_classes, last_checkpoint, ctx)
    net.hybridize()

    # Loss and metrics
    arc = Arcface_loss()
    accuracy = mx.metric.Accuracy(train_log_dir)
    eval_accuracy = mx.metric.Accuracy(eval_log_dir)

    # Summary Writer
    train_writer = SummaryWriter(train_log_dir, verbose=False)
    eval_writer = SummaryWriter(eval_log_dir, verbose=False)

    # Learning rate decay
    decay_step, decay_rate = int(args.decay_params[0]), args.decay_params[1]
    schedule = mx.lr_scheduler.FactorScheduler(decay_step, decay_rate)

    # Trainer
    optimizer_params = {'learning_rate': args.learning_rate, 'lr_scheduler': schedule}
    if args.optimizer != 'adam':
        optimizer_params['momentum'] = 0.9


    def get_trainer():
        params2opt = net.collect_params() if not args.freeze_backbone else net.collect_params('dense*')
        print(Fore.CYAN + f'{len(params2opt.keys())} params will be optimized.')
        _trainer = gluon.Trainer(
            params=params2opt,
            optimizer=args.optimizer,
            optimizer_params=optimizer_params
        )
        return _trainer


    trainer = get_trainer()

    accum_loss = 0.0  # accumulate loss initialize
    for step in tqdm(range(args.max_steps), leave=False, total=args.max_steps, desc='{},{}'.format(args.network, os.path.basename(args.output_dir))):
        step += start_step
        x, y = next(train_dataloader)
        x = x.as_in_context(ctx)
        y = nd.array(y, dtype='float32')
        y = y.as_in_context(ctx)

        with autograd.record(train_mode=True):
            fc, pred = net(x)
            arc_loss = arc(y, net.output.weight.data(), fc, args)

        arc_loss.backward()

        trainer.step(args.batch_size)

        # Compute metrics value
        soft_pred = nd.SoftmaxActivation(pred)
        pred_label = nd.argmax(soft_pred, axis=-1)
        accuracy.update(labels=y, preds=pred_label)
        temp = arc_loss.asnumpy()[0]
        accum_loss += temp  # accumulate loss value
        
        if args.eval_record and step % args.eval_freq == 0:
            eval_accuracy.reset()
            eval_mean_loss = 0
            eval_loss = []
            for x, y in tqdm(eval_dataloader, desc='Evaluating', leave=False):
                x = x.as_in_context(ctx)
                y = y.as_in_context(ctx)

                with autograd.predict_mode():
                    fc, pred = net(x)
                    arc_loss = arc(y, net.output.weight.data(), fc, args)

                # Compute metrics value
                soft_pred = nd.SoftmaxActivation(pred)
                pred_label = nd.argmax(soft_pred, axis=-1)
                eval_accuracy.update(labels=y, preds=pred_label)

                curr_loss = arc_loss.asnumpy()[0]
                eval_loss.append(curr_loss)
                
            eval_mean_acc = eval_accuracy.get()[1]
            eval_mean_loss = sum(eval_loss) / len(eval_loss)

            eval_writer.add_scalar('loss', eval_mean_loss, step)
            eval_writer.add_scalar('acc', eval_mean_acc, step)

            tqdm.write(
                Fore.GREEN +
                '[eval] step {} - loss {:.6} - acc {:.6}'.format(step, eval_mean_loss, eval_mean_acc)
            )

        if step % args.print_freq == 0:
            curr_lr = trainer.learning_rate
            curr_loss = accum_loss / args.print_freq
            curr_acc = accuracy.get()[1]

            # Write summaries
            train_writer.add_scalar('lr', curr_lr, step)
            train_writer.add_scalar('loss', curr_loss, step)
            train_writer.add_scalar('acc', curr_acc, step)

            tqdm.write(
                'step {:>5d} lr {:.1e} - loss {:.6} - acc {:.6}'.format(step, curr_lr, curr_loss, curr_acc)
            )

            accum_loss = 0.0
            accuracy.reset()
        
        
        if step % args.save_freq == 0:
            net.export(os.path.join(model_dir, 'model'), epoch=step // 100)

        if args.freeze_backbone and step >= args.freeze_steps:
            args.freeze_backbone = False
            trainer = get_trainer()