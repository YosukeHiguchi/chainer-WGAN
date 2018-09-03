import numpy as np

import chainer
from chainer import cuda
from chainer import Variable
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L


class WGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, converter=convert.concat_examples, device=None, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iterators = kwargs.pop('iterators')
        self._optimizers = kwargs.pop('optimizers')
        self.converter = converter
        self.device = device
        self.iteration = 0

        params = kwargs.pop('params')
        self.batchsize = params['batchsize']
        self.n_latent = params['n_latent']
        self.n_dis = params['n_dis']

    def update_core(self):
        gen, dis = self.gen, self.dis
        opt_gen = self._optimizers['gen']
        opt_dis = self._optimizers['dis']

        # discriminator
        for i in range(self.n_dis):
            batch = self._iterators['main'].next()
            x = Variable(self.converter(batch, self.device))
            xp = cuda.get_array_module(x.data)

            y_real = dis(x)
            z = Variable(xp.asarray(gen.make_hidden(self.batchsize)))
            x_fake = gen(z)
            y_fake = dis(x_fake)

            wasserstein_dist = F.average(y_real - y_fake)
            loss_dis = -wasserstein_dist

            dis.cleargrads()
            loss_dis.backward()
            opt_dis.update()

        # generator
        batch = self._iterators['main'].next()
        x = Variable(self.converter(batch, self.device))
        xp = cuda.get_array_module(x.data)

        z = Variable(xp.asarray(gen.make_hidden(self.batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)
        loss_gen = -F.average(y_fake)

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        chainer.reporter.report({'loss': loss_gen}, gen)
        chainer.reporter.report({'loss': loss_dis}, dis)
        chainer.reporter.report({'was. dist': wasserstein_dist})


class WGANGPUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, converter=convert.concat_examples, device=None, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iterators = kwargs.pop('iterators')
        self._optimizers = kwargs.pop('optimizers')
        self.converter = converter
        self.device = device
        self.iteration = 0

    def update_core(self):
        'hoge'
