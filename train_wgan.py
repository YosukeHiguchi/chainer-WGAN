import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
from chainer import training
from chainer import cuda
from chainer.training import extensions

import net
from updater import WGANUpdater, WGANGPUpdater
from visualize import out_generated_image


class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            xp = cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)

def main():
    parser = argparse.ArgumentParser(description='WGAN MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                    help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--out', '-o', type=str, default='model',
                        help='path to the output directory')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='dimention of encoded vector')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapepoch', '-s', type=int, default=10,
                        help='number of epochs to snapshot')
    parser.add_argument('--load_gen_model', type=str, default='',
                        help='load generator model')
    parser.add_argument('--load_dis_model', type=str, default='',
                        help='load generator model')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(args)


    gen = net.Generator(784, args.dimz, 500)
    dis = net.Discriminator(784, 500)

    if args.load_gen_model != '':
        chainer.serializers.load_npz(args.load_gen_model, gen)
    if args.load_dis_model != '':
        chainer.serializers.load_npz(args.load_dis_model, dis)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        print('GPU {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    opt_gen = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    opt_dis = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    #opt_gen = chainer.optimizers.RMSprop(5e-5)
    opt_gen.setup(gen)
    #opt_gen.add_hook(chainer.optimizer.GradientClipping(1))

    #opt_dis = chainer.optimizers.RMSprop(5e-5)
    opt_dis.setup(dis)
    opt_dis.add_hook(WeightClipping(0.01))

    train, _ = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)

    updater = WGANUpdater(
        models=(gen, dis),
        iterators={
            'main': train_iter
        },
        optimizers={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=args.gpu,
        params={
            'batchsize': args.batchsize,
            'n_latent': args.dimz,
            'n_dis': args.n_dis
        })
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.dump_graph('was. dist'))

    snapshot_interval = (args.snapepoch, 'epoch')
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(
    #     gen, 'gen{.updater.epoch}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(
    #     dis, 'dis{.updater.epoch}.npz'), trigger=snapshot_interval)

    trainer.extend(extensions.PlotReport(['loss/generator'], 'epoch', file_name='generator.png'))

    log_keys = ['epoch', 'iteration', 'was. dist', 'gen/loss', 'dis/loss']
    trainer.extend(extensions.LogReport(keys=log_keys))
    trainer.extend(extensions.PrintReport(log_keys))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(out_generated_image(gen, 20, 20, args.seed, args.out),
        trigger=(1, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    main()
