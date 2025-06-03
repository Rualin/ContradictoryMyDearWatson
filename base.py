import os
from math import ceil

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchmetrics.classification as tmc


MODEL_NAME = "MyModel"
METRICS = [
    tmc.MulticlassAccuracy(3),
    tmc.MulticlassF1Score(3)
]
METRICS_NAMES = ["Accuracy", "F1Score"]
# DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)


def compute_metrics(eval_pred):
    # accuracy_metric = load_metric("accuracy")
    # f1_metric = load_metric("f1")
    acc = tmc.MulticlassAccuracy(3, validate_args=False)
    f1 = tmc.MulticlassF1Score(3, validate_args=False)

    # print("Eval_pred:", eval_pred)
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    # print("Logits:", logits)
    # print("Labels:", labels)
    preds = np.argmax(logits, axis=-1)
    # print("Predictions:", preds)
    # print(type(preds), type(labels))
    acc.update(preds, labels)
    f1.update(preds, labels)
    metrics = {
      "accuracy" : acc.compute(),
      "f1" : f1.compute()
    }
    return {f"eval_{k}": v for k, v in metrics.items()}

def metrics_plot(**metrics):
    length = len(metrics)
    ncols = 2
    nrows = ceil(length / 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, (name, val) in enumerate(metrics.items()):
        ax[i // ncols, i % ncols].plot(val)
        ax[i // ncols, i % ncols].set_title(name)
    fig.suptitle("Metrics")
    return fig


def train_val_loop(model, criterion, optimizer, dl, is_val):
    if is_val:
        model.eval()
    else:
        model.train()
    running_loss = 0.0
    mf1s = tmc.MulticlassF1Score(17)
    macc = tmc.MulticlassAccuracy(17)
    mprec = tmc.MulticlassPrecision(17)
    mrec = tmc.MulticlassRecall(17)
    for i, data in enumerate(tqdm(dl, desc="Iterations")):
        inputs = data
        # labels = data[1].to(DEVICE)
        labels = data["labels"]
        if not(is_val):
            optimizer.zero_grad()
        outputs:torch.Tensor = model(**inputs)
        # print(outputs.shape)
        # outputs = torch.softmax(outputs, dim=1)
        # print(outputs.shape)
        loss = outputs.loss
        # loss = criterion(outputs, labels)
        if not(is_val):
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        # print(outputs)
        _, outputs = outputs.logits.max(dim=1)
        # print(outputs)
        # print(inds)
        # print(outputs.shape, labels.shape)
        # print(outputs)
        # print(labels)
        # break
        mf1s.update(outputs, labels)
        macc.update(outputs, labels)
        mprec.update(outputs, labels)
        mrec.update(outputs, labels)
    if is_val:
        print("\tValid:")
    else:
        print("\tTrain:")
    print(f'\t\tLoss: {(running_loss / len(dl)):.3f}, acc: {macc.compute():.3f}, f1: {mf1s.compute():.3f}, prec: {mprec.compute():.3f}, rec: {mrec.compute():.3f}')
    return running_loss / len(dl), macc.compute().cpu(), mf1s.compute().cpu()


def training(model, criterion, lrs, train_dl, val_dl, epoches=10):
    '''
    Returns train_losses, val_losses, train_f1, val_f1, train_acc, val_acc
    '''
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accs = []
    val_accs = []
    maxf1 = 0
    lr_id = 0
    cnt_small = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrs[lr_id])
    if not(os.path.exists("saves")):
        os.mkdir("saves")
    if not(os.path.exists("plots")):
        os.mkdir("plots")
    for epoch in range(epoches):
        if (epoch % 10 == 0) and (epoch != 0):
            lr_id = min(len(lrs) - 1, lr_id + 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[lr_id])
        print("Epoch:", epoch)
        print("LR:", lrs[lr_id])
        tr_loss, tr_acc, tr_f1 = train_val_loop(model, criterion, optimizer, train_dl, False)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        train_f1s.append(tr_f1)

        val_loss, val_acc, val_f1 = train_val_loop(model, criterion, optimizer, val_dl, True)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        # break

        if val_f1 > maxf1:
            if maxf1 != 0:
                os.system("rm -f " + os.path.join("saves", f"maxf1_{(maxf1 * 100):.0f}.pth"))
            maxf1 = val_f1
            torch.save(model.state_dict(), os.path.join("saves", f"maxf1_{(val_f1 * 100):.0f}.pth"))
        if epoch % 2 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join("saves", f"epoch_{epoch}.pth"))
            names = ["TrainLosses", "ValLosses", "TrainF1", "ValF1", "TrainAcc", "ValAcc"]
            mets = dict(zip(names, [train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs]))
            plot = metrics_plot(**mets)
            plot.savefig(os.path.join("plots", f"Metrics{MODEL_NAME}_{epoch}Epoches.png"))
        if tr_loss < 0.005:
            cnt_small += 1
        else:
            cnt_small = 0
        if cnt_small == 3:
            torch.save(model.state_dict(), os.path.join("saves", f"epoch_{epoch}_small.pth"))
            names = ["TrainLosses", "ValLosses", "TrainF1", "ValF1", "TrainAcc", "ValAcc"]
            mets = dict(zip(names, [train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs]))
            plot = metrics_plot(**mets)
            plot.savefig(os.path.join("plots", f"Metrics{MODEL_NAME}_{epoch}Epoches_small.png"))
            break
    return train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs

if __name__ == "__main__":
    print("It is not main!!!")
