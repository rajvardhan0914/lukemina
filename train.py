import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from configs.config import *
from variliteformer.datasets.leukemia_dataset import get_dataloaders
from variliteformer.models.resnet_transformer import ResNetTransformer


def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader,val_loader=get_dataloaders(DATASET_PATH,IMG_SIZE,BATCH_SIZE)

    model=ResNetTransformer(MODEL_BACKBONE,NUM_CLASSES).to(device)

    # Safe loss definition (fixes weight mismatch)
    if NUM_CLASSES==2:

        class_weights=torch.tensor([1.0,1.4]).to(device)

        criterion=torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1
        )

    else:

        criterion=torch.nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    scaler=torch.amp.GradScaler(device.type)

    best_acc=0

    train_losses=[]
    val_losses=[]
    precisions=[]
    recalls=[]
    f1s=[]
    accuracies=[]

    os.makedirs(OUTPUT_DIR+"/graphs",exist_ok=True)
    os.makedirs(CHECKPOINT_DIR,exist_ok=True)

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss=0

        for imgs,labels in tqdm(train_loader,desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):

            imgs,labels=imgs.to(device),labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):

                out=model(imgs)

                loss=criterion(out,labels)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss+=loss.item()

        train_loss=running_loss/len(train_loader)
        train_losses.append(train_loss)

        model.eval()

        preds=[]
        targets=[]
        val_loss=0

        with torch.no_grad():

            for imgs,labels in val_loader:

                imgs,labels=imgs.to(device),labels.to(device)

                out=model(imgs)

                loss=criterion(out,labels)
                val_loss+=loss.item()

                p=torch.argmax(out,1)

                preds.extend(p.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        val_loss/=len(val_loader)
        val_losses.append(val_loss)

        precision=precision_score(targets,preds)
        recall=recall_score(targets,preds)
        f1=f1_score(targets,preds)

        acc=(torch.tensor(preds)==torch.tensor(targets)).float().mean().item()

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(acc)

        print(epoch+1,acc,f1)

        if acc>best_acc:

            best_acc=acc

            torch.save(
                model.state_dict(),
                f"{CHECKPOINT_DIR}/best_{MODEL_BACKBONE}.pth"
            )

        scheduler.step()

    # Graph generation (UNCHANGED)

    plt.figure()
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{OUTPUT_DIR}/graphs/loss.png")

    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{OUTPUT_DIR}/graphs/accuracy.png")

    plt.figure()
    plt.plot(precisions)
    plt.title("Precision")
    plt.savefig(f"{OUTPUT_DIR}/graphs/precision.png")

    plt.figure()
    plt.plot(recalls)
    plt.title("Recall")
    plt.savefig(f"{OUTPUT_DIR}/graphs/recall.png")

    plt.figure()
    plt.plot(f1s)
    plt.title("F1 Score")
    plt.savefig(f"{OUTPUT_DIR}/graphs/f1.png")


if __name__=="__main__":
    main()