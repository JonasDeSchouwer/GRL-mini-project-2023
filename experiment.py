"""
This file contains the configurations for the main experiment and the training loop of the experiment itself
"""

from typing import List
import math

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from gatv2 import Architecture, GATv2_1L, GATv2_2L
from dataset import IdDataset, Graph


dataset = IdDataset(load_default=True)


class Configs:
    """
    ## Configurations
    Mostly the same hyperparameters were taken as in the Shaked Brody's GATv2 experiment on Cora, to be found on https://nn.labml.ai/graphs/gatv2/
    """
    in_features = 5
    n_hidden = 5
    n_classes = 5
    n_heads = 1     # use only one head for this experiment
    dropout = 0     # use no dropout because the degrees in IdDataset are way lower than in Cora
    max_epochs = 500
    patience = 10
    loss_func = 'CrossEntropy'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = 'Adam'
    learning_rate = 5e-3
    weight_decay = 5e-4
    share_weights = False
    batch_size = 15 # chosen for this experiment


def evaluate(model: nn.Module, set: List[Graph], loss_func=None):
    model.eval()

    loss = 0
    n_accurate = 0
    n_nodes = 0
    for graph in set:

        outputs = model(graph.features, graph.adj)
        if loss_func is not None:
            loss += loss_func(outputs, graph.targets).item()

        predictions = torch.argmax(outputs, dim=1)
        n_nodes += len(predictions)
        n_accurate += torch.sum(predictions == graph.targets)

    return {
        "loss": loss,
        "acc": n_accurate/n_nodes
    }


def train(model: nn.Module, conf):
    # define optimizer
    if conf.optimizer == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(),
            lr=conf.learning_rate,
            weight_decay=conf.weight_decay,
        )
    else:
        raise Exception(f"optimizer {conf.optimizer} unknown")
    
    # SummaryWriter for logging metrics
    writer = SummaryWriter()
    
    # define loss function
    if conf.loss_func == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    else:
        raise Exception(f"loss function {conf.loss_func} unknown")

    for epoch in range(conf.max_epochs):
        model.train()

        running_loss = 0
        best_val_loss_epoch = 0
        best_val_loss_value = np.inf

        for batch in dataset.iter_batches(batch_size=conf.batch_size):
            
            optim.zero_grad()

            loss = torch.tensor(0).float()
            for graph in batch:
                outputs = model(graph.features, graph.adj)
                loss += loss_func(outputs, graph.targets)
                running_loss += loss.item()
            loss /= len(batch)
            loss.backward()
            optim.step()

        # Log metrics
        model.eval()
        train_eval = evaluate(model, dataset.training_graphs, loss_func=loss_func)
        val_eval = evaluate(model, dataset.validation_graphs, loss_func=loss_func)
        test_eval = evaluate(model, dataset.test_graphs, loss_func=loss_func)
        writer.add_scalar('Loss/train', train_eval["loss"], epoch)
        writer.add_scalar('Loss/val', val_eval["loss"], epoch)
        writer.add_scalar('Loss/test', test_eval["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_eval["acc"], epoch)
        writer.add_scalar('Accuracy/val', val_eval["acc"], epoch)
        writer.add_scalar('Accuracy/test', test_eval["acc"], epoch)

        # do early stopping based on the validation loss
        if val_eval["loss"] < best_val_loss_value:
            best_val_loss_value = val_eval["loss"]
            best_val_loss_epoch = epoch
        if best_val_loss_epoch + conf.patience <= epoch:
            print(f"EARLY STOPPING at epoch {epoch}: validation loss did not improve for {conf.patience} epochs")
            break
        
        if epoch % math.ceil(conf.max_epochs/100) == 0:
            print (f"epoch {epoch}: {running_loss}")

    writer.close()

def main():
    # Create configurations
    conf = Configs()
    # Create GATv2 models
    model_1A = GATv2_1L(in_features=conf.in_features, n_classes=conf.n_classes, n_heads=conf.n_heads, dropout=conf.dropout, share_weights=conf.share_weights, architecture=Architecture.A)
    model_1B = GATv2_1L(in_features=conf.in_features, n_classes=conf.n_classes, n_heads=conf.n_heads, dropout=conf.dropout, share_weights=conf.share_weights, architecture=Architecture.B)
    model_2A = GATv2_2L(in_features=conf.in_features, n_hidden=conf.n_hidden, n_classes=conf.n_classes, n_heads=conf.n_heads, dropout=conf.dropout, share_weights=conf.share_weights, architecture=Architecture.A, activation=nn.Tanh())
    model_2B = GATv2_2L(in_features=conf.in_features, n_hidden=conf.n_hidden, n_classes=conf.n_classes, n_heads=conf.n_heads, dropout=conf.dropout, share_weights=conf.share_weights, architecture=Architecture.B, activation=nn.Tanh())

    train(model_1A, conf)


#
if __name__ == '__main__':
    main()