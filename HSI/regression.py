import torch
import numpy as np

class weightedHuberLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()

        self.beta = beta
    def forward(self, inputs, targets, weights):
        l1_loss = torch.abs(inputs - targets)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

def reg_map_collate_fn(batch):
    '''Map collation includes map inds
    '''
    datapoints = [datapoint for datapoint,_,_ in batch]
    datapoints = torch.stack(datapoints)
    targets = torch.Tensor([target for _,target,_ in batch])
    inds = [ind for _,_,ind in batch]
    return datapoints, targets, inds

def reg_collate_fn(batch):
    '''Regression: target labels are float
    '''
    images = [image for image, _ in batch]
    images = torch.stack(images)
    targets = torch.Tensor([target for _, target in batch])
    return images, targets        

class normalizeOxygenLabels(torch.nn.Module):
    def __init__(self,
                data_props: dict={
                    'mean':8.45,
                    'std':0.092,
                    'min':7.38,
                    'max':9.67,
                    'clip_min':8.1,
                    'clip_max':8.7,
                    'strict_min':8.34,
                    'strict_max':8.64
                },
                value_transformation: str='none',
                normalization: str='none') :
        '''Normalizes SO2 values; returns (labels, normalized_lables)
        normalization = 'none','minmax','mean','scaledmax','strict'
        value_transformation=None (12+logOH), ppm
        '''
        super(normalizeOxygenLabels, self).__init__()
        assert normalization in ['none','minmax','mean','scaledmax','strict']
        assert value_transformation in ['none', 'ppm']
        self.data_props = data_props
        self.normalization = normalization
        self.value_transformation = value_transformation
        self._set_normalization_values()
    
    def _set_normalization_values(self):
        """sets mean, std, clip"""
        self.mean=0.0
        self.std=1.0
        self.clip=False
        self.clip_min = self.data_props['clip_min']
        self.clip_max = self.data_props['clip_max']
        
        if self.normalization == 'minmax':
            self.mean = self.data_props['min']
            self.std = self.data_props['max'] - self.data_props['min']
        elif self.normalization == 'mean':
            self.mean = self.data_props['mean']
            self.std = self.data_props['std']
        elif self.normalization == 'scaledmax':
            self.mean = self.data_props['clip_min']
            self.std = self.data_props['clip_max'] - self.data_props['clip_min']
            self.clip=True
        elif self.normalization == 'strict':
            self.mean = self.data_props['strict_min']
            self.std = self.data_props['strict_max'] - self.data_props['strict_min']
            self.clip_min = self.data_props['strict_min']
            self.clip_max = self.data_props['strict_max']
            self.clip=True
        elif self.normalization == 'clip_only':
            self.clip = True
        return
    def forward(self, input):
        """input values, output (values, normalized_values)"""
        input = np.array(input)
        if self.value_transformation == 'ppm':
            input = pow(10, input-6.0)
        if self.clip:
            input = np.clip(input, self.clip_min, self.clip_max)
        # remember, default is: (input - 0)/1.0
        normalized_input = (input - self.mean) / self.std
        #normalized_input = torch.Tensor(normalized_input)
        return normalized_input
    
    def unnormalize(self, input):
        output = (input * self.std) + self.mean
        return output