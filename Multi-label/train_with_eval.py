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

from data import ImageRecordDataset
from metric import Accuracy, F1
from network import DRML, PretrainedModel, VGGFace

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


class RandomCrop(gluon.Block):
    def __init__(self, size, interp=2):
        super(RandomCrop, self).__init__()
        if isinstance(size, collections.Iterable):
            assert len(size) == 2
            self.size = size
        else:
            self.size = (size, size)
        self.interp = interp

    def forward(self, x):
        return mx.image.random_crop(x, self.size, self.interp)[0]


class TransposeChannels(gluon.HybridBlock):
    def __init__(self):
        super(TransposeChannels, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.reverse(x, axis=0)


class Transpose(gluon.HybridBlock):
    def __init__(self):
        super(Transpose, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.cast(F.transpose(x, (2, 0, 1)), dtype='float32')


def balance_sampler(samples):
    """ignore extra negative samples to keep batch balance"""
    num_pos = nd.sum(samples == 1, axis=0)
    num_neg = nd.sum(samples == 0, axis=0)
    drop_prob = (num_neg - num_pos) / num_neg
    drop_prob = nd.where(nd.lesser(drop_prob, 0), nd.zeros_like(drop_prob), drop_prob)
    mask = nd.where(nd.greater(nd.random.uniform(0, 1, shape=samples.shape, ctx=samples.context), drop_prob),
                    nd.ones_like(samples), nd.zeros_like(samples))
    mask = nd.where(nd.equal(samples, 1), samples, mask)
    return mask


def build_network(name, classes, checkpoint=None, ctx=mx.cpu(), **kwargs):
    if name == 'drml':
        grid = kwargs.get('grid', (8, 8))
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            net = DRML(classes, grid)
            net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    elif name == 'r50':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-r50-am-lfw',
                                       'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-r50-am-lfw',
                                       'model-0000.params')
            net = PretrainedModel(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    elif name == 'mobileface':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', '..', 'model',
                                       'model-y1-test2', 'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', '..', 'model',
                                       'model-y1-test2', 'model-0000.params')
            net = PretrainedModel(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    elif name == 'vggface2':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-r50-vggface2',
                                       'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-r50-vggface2',
                                       'model-0000.params')
            net = VGGFace(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    elif name == 'dpn68':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-dpn68-vggface2', 'new',
                                       'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-dpn68-vggface2', 'new',
                                       'model-0009.params')
            net = VGGFace(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    elif name == 'd121':
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            symbol_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-d121-vggface2', 'new',
                                       'model-symbol.json')
            params_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model-d121-vggface2', 'new',
                                       'model-0000.params')
            net = VGGFace(classes, symbol_file, params_file, ctx)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net

    else:
        if checkpoint:
            print(Fore.GREEN + 'Restoring params from checkpoint: {}'.format(os.path.basename(checkpoint)))
            symbol_file = checkpoint[:-11] + 'symbol.json'
            net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
        else:
            net = vision.get_model(name=name, classes=classes)
            net.features.load_parameters(model_store.get_model_file(name), ctx=ctx, ignore_extra=True)
            net.output.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return net


def get_transforms(name):
    if name == 'drml':
        train_transform = T.Compose([T.RandomResizedCrop(170, (0.9, 1), (1, 1)), T.RandomFlipLeftRight(), T.ToTensor(),
                                     T.Normalize(0.5, 0.5)])
        eval_transform = T.Compose([T.Resize(170), T.ToTensor(), T.Normalize(0.5, 0.5)])
    elif name in ['r50', 'mobileface', 'dpn68', 'd121']:
        train_transform = T.Compose([T.RandomFlipLeftRight(), Transpose()])
        eval_transform = T.Compose([Transpose()])
    elif name == 'vggface2':
        train_transform = T.Compose([T.RandomResizedCrop(224, (0.9, 1), (1, 1)), T.RandomFlipLeftRight(), Transpose(),
                                     TransposeChannels(), T.Normalize((91.4953, 103.8827, 131.0912), (1., 1., 1.))])
        eval_transform = T.Compose([T.Resize(224), Transpose(),
                                    TransposeChannels(), T.Normalize((91.4953, 103.8827, 131.0912), (1., 1., 1.))])
    elif name == 'inceptionv3':
        train_transform = T.Compose([T.RandomResizedCrop(299, (0.9, 1), (1, 1)), T.RandomFlipLeftRight(), T.ToTensor(),
                                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        eval_transform = T.Compose(
            [T.Resize(299), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        raise ValueError("Invalid Network Input")
    return train_transform, eval_transform


if __name__ == '__main__':
    num_classes = 12
    aus = [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]

    parser = argparse.ArgumentParser()

    parser.add_argument('--record', '-rec', nargs='+', required=True, help='path of .rec file for training')
    parser.add_argument('--eval-record', '-e', nargs='+', help='path of .rec file for evaluating')
    parser.add_argument('--output-dir', '-o', help='directory for logging and saving model')
    parser.add_argument('--save-freq', '-sf', type=int, default=500, help='save model every n steps')
    parser.add_argument('--eval-freq', '-ef', type=int, default=500, help='evaluating model every n steps')
    parser.add_argument('--print-freq', '-pf', type=int, default=50, help='print logs in terminal every n steps')

    parser.add_argument('--restart', '-r', action='store_true', help='ignore any checkpoints')
    parser.add_argument('--freeze-backbone', '-fb', action='store_true', help='freeze the params in backbone')
    parser.add_argument('--freeze-steps', '-fs', type=int, default=0, help='max steps for freezing the params')
    parser.add_argument('--gpu-device', '-gpu', type=int, required=True, nargs='+', help='specify gpu id')

    parser.add_argument('--network', '-net', default='alexnet', help='choose network')
    parser.add_argument('--max-steps', '-ms', type=int, default=10000, help='max steps for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--optimizer', '-opt', default='nag', choices=['adam', 'sgd', 'nag'], help='type of optimizer')
    parser.add_argument('--decay-params', '-dp', nargs=2, type=float, default=[500, 0.8],
                        help='decay step and decay rate of learning rate')
    parser.add_argument('--enable-balance-sampler', '-b', action='store_true',
                        help='enable balance sampler in batching during training')
    parser.add_argument('--train-au', '-au', choices=aus, type=int)

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
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.train_au:
        num_classes = 1
        au_idx = aus.index(args.train_au)
        aus = [args.train_au]

    label_names = ['AU{:0>2}'.format(i) for i in aus]
    ctx = [mx.gpu(i) for i in args.gpu_device] if args.gpu_device else mx.cpu()

    # Dataloader
    train_transform, eval_transform = get_transforms(args.network)
    train_dataloader = DataLoaderWrapper(gluon.data.DataLoader(
        ImageRecordDataset(
            args.record,
            transform=lambda x, y: (train_transform(x), y)),
        args.batch_size,
        shuffle=True,
        last_batch='rollover',
        num_workers=8,
        pin_memory=True
    ))

    if args.eval_record:
        eval_dataloader = gluon.data.DataLoader(
            ImageRecordDataset(
                args.eval_record,
                transform=lambda x, y: (eval_transform(x), y)),
            args.batch_size,
            shuffle=False,
            last_batch='keep',
            num_workers=0,
            pin_memory=False
        )

    # Restore last checkpoint
    checkpoint_files = sorted([fname for fname in os.listdir(model_dir) if fname.endswith('.params')])
    last_checkpoint = os.path.join(model_dir, checkpoint_files[-1]) if checkpoint_files else None
    start_step = int(os.path.splitext(last_checkpoint)[0][-4:]) * 100 + 1 if last_checkpoint else 1

    net = build_network(args.network, num_classes, last_checkpoint, ctx)
    net.hybridize()

    # Loss and metrics
    sigmoid_binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    accuracy = Accuracy(label_names, train_log_dir)
    f1 = F1(label_names, train_log_dir)
    eval_accuracy = Accuracy(label_names)
    eval_f1 = F1(label_names)

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
    for step in tqdm(range(args.max_steps), leave=False, total=args.max_steps,
                     desc='{},{}'.format(args.network, os.path.basename(args.output_dir))):
        step += start_step
        x, y = next(train_dataloader)
        if args.train_au:
            y = y[:, au_idx].reshape((-1, 1))
        sample_weights = nd.not_equal(y, 999)
        if args.enable_balance_sampler:
            sample_weights = sample_weights * balance_sampler(y)
        x = gluon.utils.split_and_load(x, ctx, even_split=False)
        y = gluon.utils.split_and_load(y, ctx, even_split=False)
        sample_weights = gluon.utils.split_and_load(sample_weights, ctx, even_split=False)

        with autograd.record(train_mode=True):
            logits = [net(data) for data in x]
            losses = [sigmoid_binary_cross_entropy(logit, label, sample_weight)
                      for logit, label, sample_weight in zip(logits, y, sample_weights)]
        for l in losses:
            l.backward()

        trainer.step(args.batch_size)

        # Compute metrics value
        y_ = [nd.greater_equal(logit, 0) for logit in logits]
        accuracy.update(labels=y, preds=y_)
        f1.update(labels=y, preds=y_)

        curr_loss = sum([nd.mean(l).asscalar() for l in losses]) / len(losses)
        accum_loss += curr_loss  # accumulate loss value

        if step % args.print_freq == 0:
            curr_lr = trainer.learning_rate
            curr_loss = accum_loss / args.print_freq
            curr_acc = accuracy.get_average()
            curr_f1 = f1.get_average()

            # Write summaries
            train_writer.add_scalar('lr', curr_lr, step)
            train_writer.add_scalar('loss', curr_loss, step)
            train_writer.add_scalar('acc', curr_acc, step)
            train_writer.add_scalar('f1', curr_f1, step)

            tqdm.write(
                'step {:>5d} lr {:.1e} - loss {:.6} - acc {:.6} - f1 {:.6}'.format(step, curr_lr, curr_loss, curr_acc,
                                                                                   curr_f1)
            )

            accum_loss = 0.0
            accuracy.reset()
            f1.reset()

        if args.eval_record and step % args.eval_freq == 0:
            eval_accuracy.reset()
            eval_f1.reset()
            eval_loss = []
            for x, y in tqdm(eval_dataloader, desc='Evaluating', leave=False):
                if args.train_au:
                    y = y[:, au_idx].reshape((-1, 1))
                sample_weights = nd.not_equal(y, 999)
                x = gluon.utils.split_and_load(x, ctx, even_split=False)
                y = gluon.utils.split_and_load(y, ctx, even_split=False)
                sample_weights = gluon.utils.split_and_load(sample_weights, ctx, even_split=False)

                with autograd.predict_mode():
                    logits = [net(data) for data in x]
                    losses = [sigmoid_binary_cross_entropy(logit, label, sample_weight)
                              for logit, label, sample_weight in zip(logits, y, sample_weights)]

                # Compute metrics value
                y_ = [nd.greater_equal(logit, 0) for logit in logits]
                eval_accuracy.update(labels=y, preds=y_)
                eval_f1.update(labels=y, preds=y_)

            curr_loss = sum([nd.mean(l).asscalar() for l in losses]) / len(losses)
            eval_loss.append(curr_loss)
            eval_mean_acc = eval_accuracy.get_average()
            eval_mean_f1 = eval_f1.get_average()
            eval_mean_loss = sum(eval_loss) / len(eval_loss)

            eval_writer.add_scalar('loss', eval_mean_loss, step)
            eval_writer.add_scalar('acc', eval_mean_acc, step)
            eval_writer.add_scalar('f1', eval_mean_f1, step)

            tqdm.write(
                Fore.GREEN +
                '[eval] step {} - loss {:.6} - acc {:.6} - f1 {:.6}'.format(step, eval_mean_loss, eval_mean_acc,
                                                                            eval_mean_f1)
            )

        if step % args.save_freq == 0:
            net.export(os.path.join(model_dir, 'model'), epoch=step // 100)

        # change trainer to optimize different params
        if args.freeze_backbone and step >= args.freeze_steps:
            args.freeze_backbone = False
            trainer = get_trainer()

    if args.eval_record:
        table = PrettyTable(['AU', 'Acc', 'F1'])
        table.float_format = '.3'
        for t_name, t_acc, t_f1 in zip(label_names, eval_accuracy.get()[1], eval_f1.get()[1]):
            table.add_row([int(t_name[-2:]), t_acc, t_f1])
        table.add_row(['Avg.', eval_accuracy.get_average(), eval_f1.get_average()])
        print(table)
