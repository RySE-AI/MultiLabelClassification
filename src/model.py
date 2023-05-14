import torch
import lightning.pytorch as pl
from collections import OrderedDict
from torch import optim, nn, utils, Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy

def basic_linear_block(input_size: int,
                       output_size: int,
                       dropout: float):
    linear_layer = nn.Sequential(nn.Linear(input_size, output_size),
                                 nn.Dropout(p=dropout),
                                 nn.LeakyReLU(inplace=True))

    return linear_layer

class ClassifierHead(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size_1: int,
                 hidden_size_2: int,
                 num_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        classifier_config = OrderedDict({
                "fc1": basic_linear_block(input_size, hidden_size_1, dropout),
                "fc2": basic_linear_block(hidden_size_1, hidden_size_2, dropout),
                "fc3": nn.Linear(hidden_size_2, num_classes)})

        self.classifier = nn.Sequential(classifier_config)

    def forward(self, x):
        x = self.classifier(x)
        return x

class Backbone(nn.Module):
    def __init__(self,
                 freeze_params: bool = True,
                 state_dict_path = None,
                 weights = "IMAGENET1K_V2",
                 backbone = "resnet50"):
        super().__init__()
        if state_dict_path is not None:
            weights = None

        self.freezed = freeze_params

        self.backbone = torch.hub.load('pytorch/vision',
                                       backbone,
                                       weights=weights)

        # If you pass a state_dict pth for the network it will load your weights
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            keys = self.backbone.load_state_dict(state_dict)
            print(f"Backbone: {keys}")

        # freeze the whole network
        if self.freezed:
            for param in self.backbone.parameters():
                param.requires_grad = False

        final_layer = list(self.backbone.named_modules())[-1]

        # saving the output size for the linear classifier's input init.
        self._output_layer_size = final_layer[1].in_features
        # Not sure if this will work for all backbones :D Resnet checked
        setattr(self.backbone, final_layer[0], nn.Identity())

    def forward(self, x):
        x = self.backbone(x)

        return x
    
class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, 
                 num_classes: int,
                 backbone_config: dict = dict(),
                 hidden_size_1: int = 2048,
                 hidden_size_2: int = 2048,
                 dropout: float = 0.5,
                 lr: float = 1e-3,
                 criterion = nn.BCELoss(),
                 test_metric = None):
        super().__init__()
        
        # Model Configuration
        self.backbone = Backbone(**backbone_config)
        self.classifier = ClassifierHead(self.backbone._output_layer_size,
                                         hidden_size_1=hidden_size_1,
                                         hidden_size_2=hidden_size_2,
                                         num_classes=num_classes,
                                         dropout=dropout)
        
        # Attributes
        self.lr = lr
        self.criterion = criterion
        
        self.train_metric = MultilabelAccuracy(num_classes)
        self.valid_metric = MultilabelAccuracy(num_classes)
        
        if test_metric: 
            self.test_metric = test_metric
        else:
            self.test_metric = MultilabelAccuracy(num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features) 
         
        return logits

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        
        train_loss = self.criterion(logits, targets)
        #torchmetrics uses a sigmoid function to calculate the accurracy
        self.train_metric(logits, targets)
        
        # Logging
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_metric, on_step=True, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        
        val_loss = self.criterion(logits, targets)
        self.valid_metric(logits, targets)
        
        # Logging
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        self.log('valid_acc', self.valid_metric, on_step=True, on_epoch=True)
        
        return val_loss
        
    def test_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
         
        test_loss = self.criterion(logits, targets)
        self.test_metric(logits, targets)
        
        # Logging only at epoch end
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.backbone.freezed:
            optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer