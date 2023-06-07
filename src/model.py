import torch
import lightning.pytorch as pl
from collections import OrderedDict
from torch import optim, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall

from typing import Optional


# TODO: Add Docstrings
# TODO: Check other Backbones (only checked resnet)
# TODO: xxx_step Code repetitions
def basic_linear_block(
    input_size: int, output_size: int, dropout: float
) -> nn.Sequential:
    """Return a linear building Block with dropout and Leaky Relu Activation."""

    linear_layer = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Dropout(p=dropout),
        nn.LeakyReLU(inplace=True),
    )

    return linear_layer


class ClassifierHead(nn.Module):
    """MLP Classifier Head to replace the classifier of a bacbbone like Resnet.
    The Classifier contains two hidden layers with Dropout/Leaky ReLu and the
    last layer is a simple linear layer without activation to get the logits"""

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        """Initialize a ClassifierHead Instance with following parameters:

        Args:
            input_size (int): Input Layer size, usually depending on your
            backbone
            hidden_size_1 (int): Size of the 1st hidden layer
            hidden_size_2 (int): Size of the 2nd hidden layer
            num_classes (int): Size of your output = number of classes
            dropout (float, optional): Set the dropout prob for the hidden layers. Defaults to 0.5.
        """
        super().__init__()
        classifier_config = OrderedDict(
            {
                "fc1": basic_linear_block(input_size, hidden_size_1, dropout),
                "fc2": basic_linear_block(hidden_size_1, hidden_size_2, dropout),
                "fc3": nn.Linear(hidden_size_2, num_classes),
            }
        )

        self.classifier = nn.Sequential(classifier_config)

    def forward(self, x):
        x = self.classifier(x)
        return x


class ResnetBackbone(nn.Module):
    """Head-/classifierless backbone class based on Resnet where the classifier
    is replaced with a Identity layer to extract the features of the backbone.
    """

    def __init__(
        self,
        freeze_params: bool = True,
        state_dict_path: Optional[str] = None,
        weights="IMAGENET1K_V2",
        backbone="resnet50",
    ):
        """Intitalise the Resnet

        Args:
            freeze_params (bool, optional): To freeze the backbones weights.
            Defaults to True.
            state_dict_path (str, optional): If you want to pass your own 
            trained resnet based model. Pass the .pth file. Defaults to None.
            weights (str, optional): Define the weights you want to use for your
            backbone. Defaults to "IMAGENET1K_V2".
            backbone (str, optional): Define which Resnet Backbone you want to
            use. Defaults to "resnet50".
        """
        super().__init__()
        if state_dict_path is not None:
            weights = None

        if "resnet" not in backbone:
            raise ValueError("Please use a Resnet architecture based Backbone")

        self.freezed = freeze_params
        self.backbone = torch.hub.load("pytorch/vision", backbone, weights=weights)

        # If you pass a state_dict pth for the network it will load your weights
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            keys = self.backbone.load_state_dict(state_dict)
            print(f"Backbone: {keys}")

        # freeze the whole network
        if self.freezed:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # final_layer = list(self.backbone.named_modules())[-1]
        final_layer = list(self.backbone.children())[-1]

        # saving the output size for the linear classifier's input init.
        self._output_layer_size = final_layer.in_features

        # Replace Classifier with Identiy Layer
        self.backbone.fc = nn.Identity()
        # setattr(self.backbone, final_layer[0], nn.Identity())

    def forward(self, x):
        x = self.backbone(x)

        return x


class MultiLabelClassifier(pl.LightningModule):
    """Lightning class for our experiment with a ResnetBackbone and a
    ClassifierHead. This module is specified for a multi labeling task.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_config: dict = dict(),
        hidden_size_1: int = 2048,
        hidden_size_2: int = 2048,
        dropout: float = 0.5,
        lr: float = 1e-3,
        criterion=nn.BCEWithLogitsLoss(),
        test_metrics=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])
        # Model Configuration
        self.backbone = ResnetBackbone(**backbone_config)
        self.classifier = ClassifierHead(
            self.backbone._output_layer_size,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Attributes
        self.lr = lr
        self.criterion = criterion
        
        metrics = MetricCollection([
            MultilabelAccuracy(num_classes),
            MultilabelPrecision(num_classes),
            MultilabelRecall(num_classes)
        ])

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")

        if test_metrics:
            self.test_metrics = test_metrics
        else:
            self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        train_loss = self.criterion(logits, targets)
        # torchmetrics uses a sigmoid function to calculate the accurracy
        self.train_metrics(logits, targets)

        # Logging
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        val_loss = self.criterion(logits, targets)
        self.valid_metrics(logits, targets)

        # Logging
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        test_loss = self.criterion(logits, targets)
        self.test_metrics(logits, targets)

        # Logging only at epoch end
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.backbone.freezed:
            optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
