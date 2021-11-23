import os
import sys
import time
import telepot
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
'''
Wrapper for the tensorboard summary writer which creates a directory for 
each study/run so that it is easier to display in tensorboard
'''


class CustomLogger():
    def __init__(self, log_dir='./runs/', new_dir=True, notify=False):
        if sys.gettrace() is None:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            self.log_dir = log_dir

            self.new_dir = new_dir

            if new_dir:
                name = str(datetime.fromtimestamp(time.time())).replace(
                    ' ', '_')
                self.log_dir = os.path.join(log_dir, 'study_' + name)
                self.dir = os.path.join(self.log_dir, 'runs_' + name)
                os.makedirs(self.dir)
            else:
                name = str(datetime.fromtimestamp(time.time())).replace(
                    ' ', '_')
                self.dir = os.path.join(log_dir, 'runs_' + name)

            self.writer = SummaryWriter(log_dir=self.dir)

            self.bot = BotLogger(notify)

    def new_run(self):
        assert self.new_dir, 'Logger is not setup to handle runs reseting, use new_dir=True instead.'
        name = str(datetime.fromtimestamp(time.time())).replace(' ', '_')
        self.dir = os.path.join(self.log_dir, 'runs_' + name)
        self.writer = SummaryWriter(log_dir=self.dir)

    def add_multi_scalars(self, dict_scalars, iteration, main_tag='Losses'):
        for k, v in dict_scalars.items():
            self.writer.add_scalar('{}/{}'.format(main_tag, k), v, iteration)


class BotLogger():
    def __init__(self, notify):
        self.notify = notify
        if self.notify:
            # create bot instance and log in conversation
            with open('/home/pierre/Documents/bot_id') as f:
                bot_infos = f.readlines()
            self.token, self.chat_id = bot_infos[0][:-1], int(bot_infos[1])
            self.bot = telepot.Bot(self.token)

            self.previous_loss = np.inf

            last_update = self.bot.getUpdates()
            if last_update != []:
                self.offset = last_update[-1]['update_id'] + 1
            else:
                self.offset = None

    def new_training(self, config, dataset=None):
        if self.notify:
            if dataset is not None:
                text = '\U00002699 *New training started on {}*\n_Config:_ \n{}'.format(
                    dataset.name, config)
            else:
                text = '\U00002699 *New training started*\n_Config:_ {}'.format(
                    config)
            self.bot.sendMessage(self.chat_id,
                                 text=text,
                                 parse_mode='Markdown')

    def training_end(self, map_value):
        if self.notify:
            text = '\U00002705 *Training done*\nFinal mAP value: {:.3f}'.format(
                map_value)
            self.bot.sendMessage(self.chat_id,
                                 text=text,
                                 parse_mode='Markdown')

    def log(self, epoch, n_epoch, iter_nb, total_iter, loss_value, map_value):
        if self.notify:
            if loss_value > self.previous_loss:
                arrow_code = '\U00002B06'
            else:
                arrow_code = '\U00002B07'
                self.previous_loss = loss_value
            text = '''*Epoch {}/{}*\n*Iteration {}/{}*\n_Current Loss: {:.3f} {}_\n_Lowest loss so far: {:.3f}_\nVOC mAP: {:.3f}'''\
                .format(epoch+1, n_epoch, iter_nb, total_iter, loss_value, arrow_code, self.previous_loss, map_value)
            self.bot.sendMessage(self.chat_id,
                                 text=text,
                                 parse_mode="Markdown")

    def report_error(self, error):
        if self.notify:
            cross_mark = '\U0000274C'
            text = "{} *An error occured during training* {}\n_Error message:_ {}".format(
                cross_mark, cross_mark, error)
            self.bot.sendMessage(self.chat_id,
                                 text=text,
                                 parse_mode="Markdown")

    def check_updates(self):
        if self.notify:
            update = self.bot.getUpdates(offset=self.offset)
            if update != []:
                if self.offset is None:
                    self.offset = update[-1]['update_id']
                else:
                    self.offset += 1

                if update[-1]['message']['text'] == '/stop':
                    text = '\U000026D4 * Stopping training * \U000026D4'
                    self.bot.sendMessage(self.chat_id,
                                         text,
                                         parse_mode="Markdown")
                    sys.exit('Training stopped by the bot.')
