#!/usr/bin/env python3.6

from typing import Any, List

import torch
from torch import Tensor
import pandas as pd
from utils import eq


class ConstantBounds():
    def __init__(self, **kwargs):
        self.C: int = kwargs['C']
        self.const: Tensor = torch.zeros((self.C, 1, 2), dtype=torch.float32)

        for i, (low, high) in kwargs['values'].items():
            self.const[i, 0, 0] = low
            self.const[i, 0, 1] = high

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        return self.const


class TagBounds(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0
        #weak_positive_class: Tensor = torch.einsum("cwh->c", [weak_target]) > 0
        c,w,h = target.shape
        masked_positive: Tensor = torch.einsum("c,c->c", [positive_class, self.idc_mask]).type(torch.float32)  # Keep only the idc
        #masked_weak: Tensor = torch.einsum("c,c->c", [weak_positive_class, self.idc_mask]).type(torch.float32)
        #assert eq(masked_positive, masked_weak), f"Unconsistent tags between labels: {filename}"
        if masked_positive.sum() ==0: # only background
            print("negative image",filename)
            res =  torch.zeros((self.C, 1, 2), dtype=torch.float32)
            res[0,0,1] = w*h
            res[0,0,0] = w*h
        else:    
            #print("positive image",filename)
            res: Tensor = super().__call__(image, target, weak_target, filename)
            res = torch.einsum("cki,c->cki", [res, masked_positive])
        #print(res)
        return res



class TagBoundsPos(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0
        weak_positive_class: Tensor = torch.einsum("cwh->c", [weak_target]) > 0

        masked_positive: Tensor = torch.einsum("c,c->c", [positive_class, self.idc_mask]).type(torch.float32)  # Keep only the idc
        masked_weak: Tensor = torch.einsum("c,c->c", [weak_positive_class, self.idc_mask]).type(torch.float32)
        #assert eq(masked_positive, masked_weak), f"Unconsistent tags between labels: {filename}"

        res: Tensor = super().__call__(image, target, weak_target, filename)
        masked_res = torch.einsum("cki,c->cki", [res, masked_positive])

        return masked_res


class PreciseBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            value: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res

class PreciseBoundsOnWeak():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(weak_target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            value: Tensor = self.__fn__(weak_target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res

class PreciseBoundsOnWeakWTags():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(weak_target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
            value_gt: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            value: Tensor = self.__fn__(weak_target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
            value_gt: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        if value_gt[1]*0 == value_gt[1]:
            #print("gt size is 0")
            value = value_gt
        else:
            if value[1]*0 == value[1]: 
                #print("inf size is 0")
                value[1] = torch.ones_like(value[1]).type(torch.float32)
            #else:
                #print("both are >0")
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res

class PreciseTags(PreciseBounds):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_value: List = kwargs['neg_value']

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0

        res = super().__call__(image, target, weak_target, filename)

        masked = res[...]
        masked[positive_class == 0] = torch.Tensor(self.neg_value)

        return masked


class BoxBounds():
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.einsum("cwh->c", [weak_target])[..., None].type(torch.float32)

        bounds: Tensor = box_sizes * self.margins

        res = bounds[:, None, :]
        assert res.shape == (c, 1, 2)
        assert (res[..., 0] <= res[..., 1]).all()
        return res


class PredictionBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #with torch.no_grad():
        #    value: Tensor = self.net(image[None, ...])[0].type(torch.float32)[..., None]  # cwh and not bcwh
        c,w,h=target.shape
        pred_size_col = 'val_pred_size'
        value = self.sizes.at[filename,pred_size_col]
        value = torch.tensor([w*h - value, value]).unsqueeze(1)
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PredictionBoundswTags():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)
        self.idc: List[int] = kwargs['idc']
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c, w, h = target.shape
        pred_size_col = 'val_pred_size'
        gt_size_col = "val_gt_size"
        value = self.sizes.at[filename,pred_size_col]
        value_gt = self.sizes.at[filename,gt_size_col]
#        print(value_gt)
        if value_gt == 0:
            value = value_gt
        value = torch.tensor([w*h - value, value]).unsqueeze(1)
        margin: Tensor
        if value_gt == 0:
            margin = torch.zeros_like(value)
        else:
            if self.mode == "percentage":
                margin = value * self.margin
            elif self.mode == "abs":
                margin = torch.ones_like(value) * self.margin
            else:
                raise ValueError("mode")
        #print('sizeloss',filename,value)
        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin.type(torch.float32), torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PredictionValues():
    def __init__(self, **kwargs):
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #print(weak_target.shape,'weak shape',filename)
        c,w,h=target.shape
        pred_size_col = 'val_pred_size'
        gt_size_col="val_gt_size"
        #print(self.sizes.at[filename,pred_size_col])
        value = self.sizes.at[filename,pred_size_col]
        value_gt = self.sizes.at[filename,gt_size_col]
        if value_gt ==0:
            value = value_gt
        #print("proploss",filename,value)
        value = value /(w*h)
        res = torch.tensor([1-value, value])        
        #print("res shape",res.shape)
        return res

