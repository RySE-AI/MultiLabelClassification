import torch
import lightning.pytorch as pl
from collections import OrderedDict
from torch import optim, nn, utils, Tensor

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

        self.backbone = torch.hub.load('pytorch/vision',
                                       backbone,
                                       weights=weights)

        # If you pass a state_dict pth for the network it will load your weights
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            keys = self.backbone.load_state_dict(state_dict)
            print(f"Backbone: {keys}")

        # freeze the whole network
        if freeze_params:
            for param in self.backbone.parameters():
                param.requires_grad = False

        final_layer = list(self.backbone.children())[-1]

        # saving the output size for the linear classifier's input init.
        self._output_layer_size = final_layer.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)

        return x
    

