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

            chainer.reporter.report({'loss': loss_dis}, dis)
            chainer.reporter.report({'was. dist': wasserstein_dist})

        # generator
        batch = self._iterators['main'].next()
        x = Variable(self.converter(batch, self.device))

        z = Variable(xp.asarray(gen.make_hidden(self.batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)
        loss_gen = -F.average(y_fake)

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        chainer.reporter.report({'loss': loss_gen}, gen)


class WGANGPUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, converter=convert.concat_examples, device=None, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iterators = kwargs.pop('iterators')
        self._optimizers = kwargs.pop('optimizers')
        self.converter = converter
        self.device = device
        self.iteration = 0

        params = kwargs.pop('params')
        self.batchsize = params['batchsize']
        self.n_dis = params['n_dis']
        self.lam = params['lam']

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

            eps = xp.random.uniform(0, 1, size=self.batchsize).astype(np.float32)[:, None]
            x_hat = eps * x + (1 - eps) * x_fake

            wasserstein_dist = F.average(y_real - y_fake)
            grad, = chainer.grad([dis(x_hat)], [x_hat], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = F.mean_squared_error(grad, xp.ones_like(grad.data))

            loss_dis = -wasserstein_dist + self.lam * loss_gp

            dis.cleargrads()
            loss_dis.backward()
            opt_dis.update()

            chainer.reporter.report({'was. dist': wasserstein_dist})
            chainer.reporter.report({'grad. pen': loss_gp})

        # generator
        batch = self._iterators['main'].next()
        x = Variable(self.converter(batch, self.device))

        z = Variable(xp.asarray(gen.make_hidden(self.batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)
        loss_gen = -F.average(y_fake)

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        chainer.reporter.report({'loss': loss_gen}, gen)
