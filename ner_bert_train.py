import numpy
from transformers import BertTokenizer
from transformers import BertForTokenClassification
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ignite.engine import Engine
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from ignite.engine import Events
from tqdm import tqdm
from ignite.metrics import Accuracy
import shutil
import random
import heapq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(engine, batch_data):
    data_inputs, data_labels = batch_data
    output = engine.estimator(**data_inputs, labels=data_labels)
    engine.optimizer.zero_grad()
    output.loss.backward()
    engine.optimizer.step()

    data_preds = torch.argmax(output.logits, dim=-1)
    right_token_number = torch.masked_select(data_preds == data_labels, data_labels != -100).sum()
    total_token_number = torch.masked_select(data_labels, data_labels != -100).numel()
    total_token_losses = output.loss.item() * total_token_number

    return {'total_token_losses': total_token_losses,
            'total_token_number': total_token_number,
            'right_token_number': right_token_number}

    # return {'total_token_losses': random.randint(0, 1000),
    #         'total_token_number': 1,
    #         'right_token_number': 1}

def collate_function(tokenizer, batch_data):
    data_inputs, data_labels = [], []
    for data in batch_data:
        data_inputs.append(data['sentence'])
        data_labels.append(torch.tensor(data['label'], device=device))
    data_inputs = tokenizer(data_inputs, padding='longest', add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')
    data_inputs = {key: value.to(device) for key, value in data_inputs.items()}
    data_labels = pad_sequence(data_labels, batch_first=True, padding_value=-100)
    return data_inputs, data_labels

def on_epoch_started(engine):
    engine.estimator.train()
    engine.total_token_losses = 0
    engine.total_token_number = 0
    engine.right_token_number = 0
    engine.progress = tqdm(range(engine.iter_nums), ncols=110)

def on_epoch_completed(engine):
    engine.progress.close()
    do_evaluation(engine)

def on_iteration_completed(engine):
    max_epoch = engine.state.max_epochs
    cur_epoch = engine.state.epoch
    engine.total_token_losses += engine.state.output['total_token_losses']
    engine.total_token_number += engine.state.output['total_token_number']
    engine.right_token_number += engine.state.output['right_token_number']

    desc = f'training epoch {cur_epoch:2d}/{max_epoch:2d} loss {engine.total_token_losses:12.4f} acc {engine.right_token_number/engine.total_token_number:.2f}'
    engine.progress.set_description(desc)
    engine.progress.update()

@torch.no_grad()
def do_evaluation(engine):
    engine.estimator.eval()
    progress = tqdm(range(len(engine.testloader)), ncols=110)
    max_epoch = engine.state.max_epochs
    cur_epoch = engine.state.epoch
    right_token_number = 0
    total_token_number = 0
    total_token_losses = 0
    for data_input, data_labels in engine.testloader:
        output = engine.estimator(**data_input, labels=data_labels)
        data_preds = torch.argmax(output.logits, dim=-1)
        right_token_number += torch.masked_select(data_preds == data_labels, data_labels != -100).sum()
        total_token_number += torch.masked_select(data_labels, data_labels != -100).numel()
        total_token_losses += output.loss.item() * total_token_number
        desc = f'evaluate epoch {cur_epoch:2d}/{max_epoch:2d} loss {total_token_losses:12.4f} acc {right_token_number/total_token_number:.2f}'
        progress.set_description(desc)
        progress.update()
    progress.close()

    loss = round(total_token_losses, 4)
    checkpoint = f'finish/bert/epoch-{cur_epoch}-train-{engine.total_token_losses:.4f}-{engine.right_token_number/engine.total_token_number:.2f}-test-{total_token_losses:.4f}-{right_token_number/total_token_number:.2f}'
    # 存储损失最小的3个模型
    if len(engine.checkpoints) < 3:
        engine.checkpoints.append({'checkpoint': checkpoint, 'loss': loss})
        engine.estimator.save_pretrained(checkpoint)
        engine.tokenizer.save_pretrained(checkpoint)
        return

    # 如果当前 checkpoint 损失比最大还大，则不进行存储
    engine.checkpoints = sorted(engine.checkpoints, key=lambda x: x['loss'], reverse=True)
    if loss > engine.checkpoints[0]['loss']:
        return

    # 删除损失最大的模型
    shutil.rmtree(engine.checkpoints[0]['checkpoint'])
    engine.checkpoints.pop(0)
    engine.checkpoints.append({'checkpoint': checkpoint, 'loss': loss})
    engine.estimator.save_pretrained(checkpoint)
    engine.tokenizer.save_pretrained(checkpoint)


def do_train():

    checkpoint = 'pretrained/bert-base-chinese'
    estimator =  BertForTokenClassification.from_pretrained(checkpoint, num_labels=7).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)
    traindata = load_from_disk('data/train_valid.data')
    testdata = load_from_disk('data/test.data')
    trainloader = DataLoader(traindata, batch_size=4, collate_fn=partial(collate_function, tokenizer))
    testloader = DataLoader(testdata, batch_size=16, collate_fn=partial(collate_function, tokenizer))

    trainer = Engine(train_step)
    trainer.estimator = estimator
    trainer.optimizer = optimizer
    trainer.tokenizer = tokenizer
    trainer.iter_nums = len(trainloader)
    trainer.testloader = testloader
    trainer.checkpoints = []

    trainer.add_event_handler(Events.EPOCH_STARTED, on_epoch_started)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, on_epoch_completed)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, on_iteration_completed)

    trainer.run(trainloader, max_epochs=20)


if __name__ == '__main__':
    do_train()