import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as F
import torchmetrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb.plot
import pandas as pd



torch.manual_seed(1442)

device="mps"
torch.manual_seed(1442)
layers=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

class vgg16(pytorch_lightning.LightningModule):
    def __init__(self,
                 layers=layers,
                 in_channel=1, 
                 num_classes=7, 
                 dropout=0.5,
                 lr=3e-4):
        super().__init__()

        self.in_channel=in_channel
        self.num_classes=num_classes
        self.lr=lr

        self.dropout=nn.Dropout(dropout)

        # VGG16 backbone
        self.backbone=[]
        for layer in layers:
            if layer=='M':
                self.backbone+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                self.backbone+=[nn.Conv2d(in_channel,layer,kernel_size=3,padding=1),
                                nn.BatchNorm2d(layer),
                                nn.ReLU(inplace=True)]
                in_channel=layer

        self.backbone+=[nn.AdaptiveAvgPool2d((1,1))]
        self.backbone=nn.Sequential(*self.backbone)

        # classifier
        self.classifier=nn.Linear(512,num_classes)

        # acc metric
        self.acc=torchmetrics.classification.Accuracy(task="multiclass",num_classes=num_classes)
        
        # Initialize lists to store validation outputs
        self.validation_step_outputs = []

        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.2, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,x):
        # x= a batch of data = [128,1,44,44]

        # backbone
        x=self.backbone(x)           # [128,512,1,1]
        x=x.squeeze(-1).squeeze(-1)  # [128,512]

        # dropout
        x=self.dropout(x)            # [128,512]
        # classifier
        x=self.classifier(x)         # [128,7]

        return x
    
    def training_step(self,batch,batch_idx):
        # compute loss
        x,y=batch
        preds=self.forward(x) # preds = logits
        loss=F.cross_entropy(preds,y)
        self.log("train/train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/train_acc",self.acc(preds,y),prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        # x=[128,10,1,44,44] -> take average prediction over 10 subimages
        x=x.transpose(0,1) # [10,128,1,44,44]
        loss=0.0
        preds=torch.zeros(x.size(1),self.num_classes,device=device)  # [128,7]
        for i in range(x.size(0)):
            # prediction and loss for subimage i
            preds+=self.forward(x[i])
            loss+=F.cross_entropy(preds,y)
        loss/=x.size(0)
        preds/=x.size(0)

        # logging metrics
        self.log("val/val_loss",loss,prog_bar=True, on_epoch=True)
        self.log("val/val_acc",self.acc(preds,y),prog_bar=True, on_epoch=True)
        
        # Save outputs for logging in wandb
        self.validation_step_outputs.append({"labels": y, "logits": preds})


    # create confusion matrix
    def on_validation_epoch_end(self):
        outputs=self.validation_step_outputs
        labels=torch.cat([x["labels"] for x in outputs])
        logits=torch.cat([x["logits"] for x in outputs])
        preds=torch.argmax(logits,1)
        
        # Move tensors to CPU before converting to NumPy
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()

        # 2. confusion matrix plotting with seaborn
        data=confusion_matrix(labels_cpu,preds_cpu)
        df_cm=pd.DataFrame(data,columns=np.unique(labels_cpu),index=np.unique(labels_cpu))
        df_cm.index.name="Actual"
        df_cm.columns.name="Predicted"

        # plot
        plt.figure(figsize=(7,4))
        plot=sns.heatmap(df_cm,cmap="Blues", annot=True, annot_kws={"size":16})

        # log the plot into wandb
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # Clear validation_step_outputs for next epoch
        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):

        optimizer=torch.optim.SGD(self.parameters(),
                                  lr=self.lr,
                                  momentum=0.9,
                                  weight_decay=5e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode="min",
                                                             patience=5,
                                                             factor=0.1)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                "monitor": "train/train_loss",
                                "interval": "epoch", # default
                                "frequency": 1       # default
                                }
                }

if __name__=="__main__":
    model=vgg16()
    print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")