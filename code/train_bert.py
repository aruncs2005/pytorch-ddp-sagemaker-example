import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
import ast
import itertools as itr

import time

import json
import boto3
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 64  # this is the max length of the sentence

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def _get_train_data_loader(batch_size, training_dir, is_distributed):
    logger.info("Get train data loader")

    dataset = pd.read_csv(os.path.join(training_dir, "train.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader


def _get_test_data_loader(test_batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "test.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

    return train_dataloader


def train(args):
    # PyTorch DDP cluster configuration
    # ------------------------------------------------------------    
    # rank ID within a node (unique per machine, not per cluster)
    local_rank = args.local_rank
    
    # determine global rank - rank ID within the cluster (unique per machine and per cluster)
    rank_list = [str(i) + '-' + str(j) 
                 for i, j in itr.product(range(0, args.nodes), range(0, args.gpus))]
    
    global_rank_str = str(args.node_rank) + '-' + str(local_rank)
    
    global_rank = rank_list.index(global_rank_str)
    
    print(f"GPU local rank {local_rank} on machine {args.node_rank}," +
          f"is given global rank {global_rank}")
    
    print(f'MASTER_ADDR {os.environ.get("MASTER_ADDR")}')
    print(f'MASTER_PORT {os.environ.get("MASTER_PORT")}')
    
    dist.init_process_group(args.backend, rank=global_rank, world_size=args.nodes*args.gpus)
    
    
    # set the seed for generating random numbers
  
    torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, True)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    logger.info("Starting BertForSequenceClassification\n")
    # Instantiate model
    # ------------------------------------------------------------
    
    # One replica downloads the model, the other replicas read it from hub cache
    if global_rank == 0:
        model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    dist.barrier()
        
    if global_rank != 0:
        model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
  
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

        
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
   
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    )

    logger.info("End of defining BertForSequenceClassification\n")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )

        logger.info("Average training loss: %f\n", total_loss / len(train_loader))

        test(model, test_loader, device)

    logger.info("Saving tuned model.")
    if global_rank == 0:
        model_2_save = model.module if hasattr(model, "module") else model
        model_2_save.save_pretrained(save_directory=args.model_dir)


def test(model, test_loader, device):
    model.eval()
    _, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy

    logger.info("Test set: Accuracy: %f\n", tmp_eval_accuracy)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adapt arg defaults if this script runs on SageMaker
    if "SM_HOSTS" in os.environ:
        hosts = ast.literal_eval(os.environ.get("SM_HOSTS"))
        default_nodes = len(hosts)
        default_node_rank = hosts.index(os.environ.get("SM_CURRENT_HOST"))
    else:
        default_nodes = 1
        default_node_rank = 0
    
  
    parser = argparse.ArgumentParser()
    
    
     # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=2, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=100, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")
    parser.add_argument("--gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--nodes", type=int, default=default_nodes)
    parser.add_argument("--node_rank", type=int, default=default_node_rank)    
    parser.add_argument("--local_rank", type=int)  # set by PyTorch DDP launch utility

  

    args, _ = parser.parse_known_args()
    
    torch.cuda.empty_cache()
    
    
    train(args)