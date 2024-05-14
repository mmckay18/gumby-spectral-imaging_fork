""" SimpleNet

Paper: `Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures`
    - https://arxiv.org/abs/1608.06037

@article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}

Official Caffe impl at https://github.com/Coderx7/SimpleNet
Official Pythorch impl at https://github.com/Coderx7/SimpleNet_Pytorch
Seyyed Hossein Hasanpour
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, List, Dict, Any, cast, Optional

class View(nn.Module):
    def forward(self, x):
        logging.info(f"{x.shape}")
        return x

class SimpleNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        in_chans: int = 2040,
        drop_rates: Dict[int, float] = {},
    ):
        """Instantiates a SimpleNet model. SimpleNet is comprised of the most basic building blocks of a CNN architecture.
        It uses basic principles to maximize the network performance both in terms of feature representation and speed without
        resorting to complex design or operators. 
        
        Args:
            num_classes (int, optional): number of classes. Defaults to 8.
            in_chans (int, optional): number of input channels. Defaults to 2040.
            drop_rates (Dict[int,float], optional): custom drop out rates specified per layer. 
                each rate should be paired with the corrosponding layer index(pooling and cnn layers are counted only). Defaults to {}.
        """
        super(SimpleNet, self).__init__()
        # (channels or layer-type, stride=1, drp=0.)
        # everything has kernel_size=3 & padding=1 unless noted
        self.cfg: List[Tuple[Union(int, str), int, Union(float, None), Optional[str]]] = [
            (64, 1, 0.0), # Conv2D + BatchNorm2D + ReLU
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            ("p", 2, 0.0), # MaxPool2D (kernel_size=(2,2)) + dropout
            (256, 1, 0.0),
            (256, 1, 0.0),
            (256, 1, 0.0),
            (512, 1, 0.0),
            ("p", 2, 0.0), # MaxPool2D (kernel_size=(2,2)) + dropout
            (2048, 1, 0.0, "k1"), # kernel_size=1
            (256, 1, 0.0, "k1"), # kernel_size=1
            (256, 1, 0.0),
        ]
        # make sure values are in proper form
        self.dropout_rates = {int(key): float(value) for key, value in drop_rates.items()}
        # 15 is the last layer of the network(including two previous pooling layers)
        # basically specifying the dropout rate for the very last layer to be used after the pooling
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)
        self.strides = {}

        self.num_classes = num_classes
        self.in_chans = in_chans

        self.features = self._make_layers()
        self.classifier = nn.Linear(round(self.cfg[-1][0]), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = F.dropout2d(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers: List[nn.Module] = []
        input_channel = self.in_chans
        stride_list = self.strides
        for idx, (layer, stride, defaul_dropout_rate, *layer_type) in enumerate(self.cfg):
            stride = stride_list[idx] if len(stride_list) > idx else stride
            # check if any custom dropout rate is specified
            # for this layer, note that pooling also counts as 1 layer
            custom_dropout = self.dropout_rates.get(idx, None)
            custom_dropout = defaul_dropout_rate if custom_dropout is None else custom_dropout
            # dropout values must be strictly decimal. while 0 doesnt introduce any issues here
            # i.e. during training and inference, if you try to jit trace your model it will crash
            # due to using 0 as dropout value(this applies up to 1.13.1) so here is an explicit
            # check to convert any possible integer value to its decimal counterpart.
            custom_dropout = None if custom_dropout is None else float(custom_dropout)
            kernel_size = 3
            padding = 1 
            if layer_type == ['k1']:
                kernel_size = 1
                padding = 0 

            if layer == "p":
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(stride, stride)),
                    nn.Dropout2d(p=custom_dropout, inplace=True),
                ]
            else:
                filters = round(layer)
                if custom_dropout is None:
                    layers += [
                        nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(p=custom_dropout, inplace=False),
                    ]

                input_channel = filters

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        return model

class SpectraNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        in_chans: int = 1,
        drop_rates: Dict[int, float] = {},
    ):
        """Modified SimpleNet model for single pixel inputs with Conv1D.
        Expects inputs: (batch_size, in_chan, spectral_dim)
        for batch_size=2, spectral_dim=2040:
            input: [2,1,2040]
            after 1st Conv1D: [2, 64, 2040]
            after ... Conv1D: [2, 128, 2040]
            after 1st Pool1D: [2, 128, 1020]
                  ... Conv1D: [2, 512, 1020]
            after 2nd Pool1D: [2, 512, 510]
        """
        super(SpectraNet, self).__init__()
        # (channels or layer-type, stride=1, drp=0.)
        self.cfg: List[Tuple[Union(int, str), int, Union(float, None), Optional[str]]] = [
            (64, 1, 0.0), # Conv1D + BatchNorm1D + ReLU
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            ("p", 2, 0.0), # MaxPool1D + dropout
            (256, 1, 0.0),
            (256, 1, 0.0),
            (256, 1, 0.0),
            (512, 1, 0.0),
            ("p", 2, 0.0), # MaxPool1D + dropout
            (2048, 1, 0.0, "k1"), # kernel_size=1
            (256, 1, 0.0, "k1"), # kernel_size=1
            (256, 1, 0.0),
        ]
        # make sure values are in proper form
        self.dropout_rates = {int(key): float(value) for key, value in drop_rates.items()}
        # 15 is the last layer of the network (including two previous pooling layers)
        # basically specifying the dropout rate for the very last layer to be used after the pooling
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)
        self.strides = {}

        self.num_classes = num_classes
        self.in_chans = in_chans

        self.features = self._make_layers()
        self.classifier = nn.Linear(round(self.cfg[-1][0]), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.max_pool1d(out, kernel_size=out.size(-1)) # referencing channel dim
        out = F.dropout1d(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers: List[nn.Module] = []
        input_channel = self.in_chans
        stride_list = self.strides
        for idx, (layer, stride, defaul_dropout_rate, *layer_type) in enumerate(self.cfg):
            stride = stride_list[idx] if len(stride_list) > idx else stride
            # check if any custom dropout rate is specified
            # for this layer, note that pooling also counts as 1 layer
            custom_dropout = self.dropout_rates.get(idx, None)
            custom_dropout = defaul_dropout_rate if custom_dropout is None else custom_dropout
            # dropout values must be strictly decimal. while 0 doesnt introduce any issues here
            # i.e. during training and inference, if you try to jit trace your model it will crash
            # due to using 0 as dropout value(this applies up to 1.13.1) so here is an explicit
            # check to convert any possible integer value to its decimal counterpart.
            custom_dropout = None if custom_dropout is None else float(custom_dropout)
            kernel_size = 3
            padding = 1 
            if layer_type == ['k1']:
                kernel_size = 1
                padding = 0 

            if layer == "p":
                layers += [
                    nn.MaxPool1d(kernel_size=2, stride=stride),
                    nn.Dropout1d(p=custom_dropout, inplace=True),
                ]
            else:
                filters = round(layer)
                if custom_dropout is None:
                    layers += [
                        nn.Conv1d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm1d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        nn.Conv1d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm1d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout1d(p=custom_dropout, inplace=False),
                    ]

                input_channel = filters

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        return model

if __name__ == "__main__":
    num_classes = 8
    spectral_dim = 2040
    patch_size=7
    batch_size=1
    # Conv2D version
    model = SimpleNet(num_classes=num_classes, in_chans=spectral_dim)
    input_dummy = torch.randn(size=(batch_size, spectral_dim, patch_size, patch_size)) # (1,2040,7,7)
    out = model(input_dummy)
    logging.info(model)
    logging.info(f"output: {out.size()}") # (1,8)
    # Conv1D version
    model = SpectraNet(num_classes=num_classes, in_chans=1)
    input_dummy = torch.randn(size=(batch_size, 1, spectral_dim)) # (1,1,2040)
    out = model(input_dummy)
    logging.info(model)
    logging.info(f"output: {out.size()}") # (1,8)