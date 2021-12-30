import os, json, logging, collections
from copy import deepcopy
import torch, numpy as np

from datasets.loader import get_dataset
from torch.utils.data import DataLoader

from utils.utils import AverageMeter, TimeMeter
from models_ import simul_lr
from tensorboardX import SummaryWriter


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self._build_loader()
        self._build_model()
        self._build_optimizer()
        self.loss_meter = collections.defaultdict(lambda: AverageMeter())
        self.time_meter = TimeMeter()

    def train(self):
        args = self.args
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        self.tb_writer = SummaryWriter('./tensorboard')

        for epoch in range(1, self.args.max_num_epochs + 1):
            logging.info('Start Epoch {}'.format(epoch))

            loss_meter = self._train_one_epoch(epoch)
            self.tb_writer.add_scalar(self.args.mn + '_ctc_loss', loss_meter['ctc_loss'].avg, epoch)
            self.tb_writer.add_scalar(self.args.mn + '_rnnt_loss', loss_meter['rnnt_loss'].avg, epoch)

            path = os.path.join(args.model_saved_path, 'model-%s' % (args.mn))
            torch.save(self.model.state_dict(), path)
            logging.info('model saved to %s' % path)
            meters = self.eval(dec=args.dec, epoch=epoch)

        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        self.model.train()
        for bid, batch in enumerate(self.train_loader, 1):
            for k in batch.keys():
                batch[k] = batch[k].cuda(non_blocking=True)
            output = self.model(**batch, train=True, epoch=epoch)
            loss, ar_loss, ctc_loss, rnnt_loss = output['loss'].mean(), output['ar_loss'].mean(), output['ctc_loss'].mean(), output['rnnt_loss'].mean()
            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            self.loss_meter['ctc_loss'].update(ctc_loss.item())
            self.loss_meter['rnnt_loss'].update(rnnt_loss.item())
            self.loss_meter['loss'].update(loss.item())

            self.time_meter.update()
            loss_meter_ = deepcopy(self.loss_meter)
            if bid % self.args.display_n_batches == 0 or bid == self.train_loader.__len__():
                self.print_log(epoch, bid, curr_lr)

        return loss_meter_

    def eval(self, dec, epoch=None):
        data_loaders = [self.val_loader, self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        self.model.eval()

        with torch.no_grad():
            for data_loader in data_loaders:
                if data_loader == None: continue
                for bid, batch in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()
                    if self.args.device == 'gpu':
                        for k in batch.keys():
                            if k != 'y':
                                batch[k] = batch[k].cuda(non_blocking=True)
                    AL = 0
                    if 'rnnt' in dec:
                        output = self.model(**batch, train=False)
                        loss, rnnt_pred = output['loss'], output['rnnt_pred']
                        AL = np.array(output['ALs']).mean()
                        if self.args.device == 'gpu':
                            cer, wer = self.model.module.evaluate(rnnt_pred, batch['y'])
                        elif self.args.device == 'cpu':
                            cer, wer = self.model.evaluate(rnnt_pred, batch['y'])
                        meters['CER'].update(cer, len(rnnt_pred))
                        meters['WER'].update(wer, len(rnnt_pred))
                        meters['AL'].update(AL, len(rnnt_pred))

                    if self.args.evaluate:
                        print('loss:%.4f\tcer:%.4f\twer:%.4f\tAL:%.4f' % (loss, cer, wer, AL))
                        print('loss:%.4f\tcer:%.4f\twer:%.4f\tAL:%.4f' % (loss, meters['CER'].avg, meters['WER'].avg, meters['AL'].avg))

                meters_ = deepcopy(meters)
                print('| ', end='')
                for key, value in meters.items():
                    try:
                        self.tb_writer.add_scalar(self.args.mn + '_' + key, value.avg, epoch)
                    except Exception as E:
                        print(E)
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    meters[key].reset()
                print()
        return meters_

    def print_log(self, epoch, bid, curr_lr):
        msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
        for k, v in self.loss_meter.items():
            msg += '{} = {:.4f}, '.format(k, v.avg)
            v.reset()
        msg += '{:.3f} seconds/batch'.format(1.0 / self.time_meter.avg)
        logging.info(msg)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info('model saved to %s' % path)

    def _build_loader(self):
        args = self.args
        with open(args.vocab_path) as f:
            self.args.vocab = json.load(f)
        train = get_dataset(args.train_data, args)
        val = None
        test = get_dataset(args.test_data, args)
        print('trainset:%d, testset:%d\n' % (len(train), len(test)))
        self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=10,
                                       shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=0,
                                     shuffle=False) if val else None
        self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=10,
                                      shuffle=False, pin_memory=True) if test else None

    def _build_model(self):
        if self.args.model_name == 'simul_lr':
            self.model = simul_lr.Model(self.args)
        device_ids = [i for i in range(len(self.args.g.split(',')))]
        if self.args.device == 'gpu':
            self.model = self.model.to(torch.device('cuda:%d' % device_ids[0]))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if self.args.load_pretrained_model:
            if self.args.partial_load:
                old_state_dict = self.model.state_dict()  # now
                new_state_dict = torch.load(self.args.model_state_dict_path)  # load
                cnt = 0
                for k in new_state_dict.keys():
                    if k in old_state_dict.keys() and 'visual_front_encoder' in k:
                        cnt += 1
                        old_state_dict[k] = new_state_dict[k]
                for k, v in self.model.named_parameters():
                    if 'visual_front_encoder' in k:
                        v.requires_grad = False
                print('\nload', cnt, 'keys from:%s\n' % self.args.model_state_dict_path)
                self.model.load_state_dict(old_state_dict)
            else:
                if self.args.device == 'gpu':
                    self.model.load_state_dict(torch.load(self.args.model_state_dict_path), strict=True)
                elif self.args.device == 'cpu':
                    m = torch.load(self.args.model_state_dict_path)
                    m = {k.replace('module.', ''): v for k, v in m.items()}
                    self.model.load_state_dict(m, strict=False)

                print('\nload from:%s\n' % self.args.model_state_dict_path)

    def _build_optimizer(self):
        from optimizer.adam_optimizer import AdamOptimizer
        from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
        from optimizer.lr_scheduler.cosine_lr_scheduler import CosineSchedule

        l_p = list(self.model.parameters())
        self.optimizer = AdamOptimizer(self.args, l_p)
        if self.args.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineSchedule(self.args, self.optimizer)
        elif self.args.lr_scheduler == 'inverse_sqrt':
            self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)