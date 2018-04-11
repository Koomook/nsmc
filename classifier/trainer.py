import tensorflow as tf
import numpy as np

import os, json
import logging
from datetime import datetime
from tensorflow.python.tools import freeze_graph


class StepBasedTrainer():
    """A optimizer that takes batch generator

    Attibutes:
        optimizer: A optimizer produced by a optimizer
        loss: model's loss equation
        placeholders: model's input placeholders
        learning_rate: a tensor holding current learning rate (produced by optimizer)
        batch_size: number of examples to train per one step
        checkpoint_dir: directory to store tensorflow checkpoints
        model_filename: filename of model in checkpoints
        checkpoint_path: filepath of tensorflow checkpoint
        saver: to save model
        session: tensorflow session
    """

    def __init__(self, 
                 optimizer, 
                 loss, 
                 acc, 
                 placeholders,
                 keep_prob_placeholder,
                 keep_prob, 
                 lr_placeholder, 
                 model_channel,
                 max_step=100000, 
                 print_step=100, 
                 save_step=1000, 
                 warmup_step=None,
                 project_name='temp', 
                 model_filename='model_ckpt'
                 ):
        """Initialize generator-based optimizer runner

        Args:
            project_name: name of your project (must be unique)
            optimizer: tensorflow optimizer
        """

        self.optimizer = optimizer
        self.loss = loss
        self.acc = acc
        self.placeholders = placeholders
        self.keep_prob_placeholder = keep_prob_placeholder
        self.keep_prob = keep_prob
        self.lr_placeholder = lr_placeholder

        self.scaled_channel = model_channel**(-0.5)
        self.max_step = max_step
        self.print_step = print_step
        self.save_step = save_step
        
        if warmup_step:
            self.warmup_step = warmup_step
        else:
            self.warmup_step = 4000 # default value is 4000 warmupstep per 100000 steps

        self.session = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=10)

        self.project_name = project_name
        self.checkpoint_dir = os.path.join('checkpoints', self.project_name)
        if not os.path.isdir(self.checkpoint_dir):
            print('new training start : {}'.format(project_name))
            os.mkdir(self.checkpoint_dir)
            tf.global_variables_initializer().run(session=self.session)
#             self.session.graph.finalize()
            self.step = 1
        else:
            try:
                latest_path = tf.train.latest_checkpoint(self.checkpoint_dir)
                self.saver.restore(self.session, latest_path)
                self.step = int(latest_path.split('-')[-1])
            except:
                print('something')
                tf.global_variables_initializer().run(session=self.session)
#                 self.session.graph.finalize()
                self.step = 1

        self.model_filename = model_filename
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_filename)

        self.loss_list = []
        self.acc_list = []

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s | %(message)s',
                            datefmt='%m-%d %H:%M:%S',
                            filename='{}/{}.log'.format(
                                self.checkpoint_dir, datetime.now()),
                            filemode='a'
                            )

        consoleHandler = logging.StreamHandler(os.sys.stdout)
        consoleTitle = logging.StreamHandler()
        logging.getLogger().addHandler(consoleHandler)

        self.logger = logging.getLogger()

    def run(self, train_batch_producer, valid_batch_producer):
        """Run optimizer with given training and validation dataset
        This prints training logs and save the model during the training

        Args:
            train_batch_producer: a generator that produces training data to feed model
            valid_batch_producer: a generator that produces valid data to feed model
        """
        print('It runs')
        lr = min(self.step**(-0.5), self.step*self.warmup_step**(-1.5))*self.scaled_channel
        while self.step < self.max_step+1:
            # set learning rate
            if self.step < self.warmup_step:
                b = self.step*(self.warmup_step**(-1.5))
            else:
                b = self.step**(-0.5)
            lr = self.scaled_channel * b

            data_batch = next(train_batch_producer)
            self.feed_dict = {}
            zipped = zip(self.placeholders, data_batch)
            self.feed_dict.update(zipped)
            self.feed_dict.update({self.keep_prob_placeholder:self.keep_prob})
            self.feed_dict.update({self.lr_placeholder:lr})
            
            _, loss_value, acc = self.session.run([self.optimizer, self.loss, self.acc], feed_dict=self.feed_dict)
            self.loss_list.append(loss_value)
            self.acc_list.append(acc)

            if self.step % self.print_step == 0:

                average_loss = np.mean(self.loss_list)
                average_acc = np.mean(self.acc_list)

                valid_batch = next(valid_batch_producer)
                valid_dict = {}
                zipped = zip(self.placeholders, valid_batch)
                valid_dict.update(zipped)
                valid_dict.update({self.keep_prob_placeholder:1.0})
                valid_loss, valid_acc = self.session.run([self.loss, self.acc], feed_dict=valid_dict)
                # Leave log
                base_message = ("Step: {step:<6d} "
                                " Training Loss: {average_loss:<.6}"
                                " Valid Loss: {valid_loss:<.6}"
                                " Training Accuracy: {average_acc:<.3}"
                                " Valid Accuracy: {valid_acc:<.3}"
                                " Learning rate: {learning_rate:<.6} ")

                message = base_message.format(
                    step=self.step,
                    average_loss=average_loss,
                    loss=loss_value,
                    acc=average_acc,
                    learning_rate=lr
                )

                self.logger.info(message)

                self.loss_list = [] # initialize
                self.acc_list = []

            # Save
            if self.step % self.save_step == 0:
                self.saver.save(self.session, self.checkpoint_path, global_step=self.step)

            self.step += 1
        print('done')