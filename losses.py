#!/usr/env/bin python3.6

from typing import List, Tuple
# from functools import reduce
from operator import add
from functools import reduce
import numpy as np
import torch
from torch import einsum
from torch import Tensor

from utils import simplex, sset, probs2one_hot
import torch.nn.modules.padding
from torch.nn import BCEWithLogitsLoss


class SelfEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        #print(probs.shape,target.shape)
        #assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        #import matplotlib.pyplot as plt
        #plt.imshow(probs[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        #plt.imshow(target[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())

        return loss


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        #print(probs.shape,target.shape)
        #assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        #import matplotlib.pyplot as plt
        #plt.imshow(probs[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        #plt.imshow(target[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        #print(loss)
        return loss


class NegCrossEntropy(CrossEntropy):
    """
    Apply the cross-entropy ONLY if the whole image is the selected class.
    This is useful to supervise the background class when we have weak labels.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        _, _, w, h = probs.shape
        trimmed: Tensor = target[:, self.idc, ...]
        full_img: Tensor = torch.einsum("bcwh->b", trimmed) == (w * h)  # List of images that are fully covered

        if full_img.any():
            where = torch.nonzero(full_img).flatten()
            return super().__call__(probs[where], target[where], bounds[where])

        return torch.zeros(1).to(probs.device)



class NaivePenalty():
    """
    Implementation in the same fashion as the log-barrier
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.power: int = kwargs["power"]
        self.curi: int = kwargs["curi"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        def penalty(z: Tensor) -> Tensor:
            assert z.shape == ()

            return torch.max(torch.zeros_like(z), z)**2

        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        #assert probs.shape == target.shape
        #print(probs.shape, target.shape)
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        _, _, k, two = bounds.shape  # scalar or vector
        #print(bounds.shape,"bounds shape")
        assert two == 2
        # assert k == 1  # Keep it simple for now
        value: Tensor = self.__fn__(probs[:, self.idc, ...],self.power)
        #print(value.shape,"value shape")

        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        if len(value.shape)==2: #then its norm soft size ... to ameleiorate
            value = value.unsqueeze(2)
            lower_b = lower_b/(w*h)
            upper_b = upper_b/(w*h)
        #    print(np.around(lower_b.cpu().numpy().flatten()), np.around(upper_b.cpu().numpy().flatten()), np.around(value.cpu().detach().numpy().flatten()))

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = (value - upper_b).type(torch.float32).flatten()
        lower_z: Tensor = (lower_b - value).type(torch.float32).flatten()

        upper_penalty: Tensor = reduce(add, (penalty(e) for e in upper_z))
        lower_penalty: Tensor = reduce(add, (penalty(e) for e in lower_z))
        #count_up: Tensor = reduce(add, (penalty(e)>0 for e in lower_z))
        #count_low: Tensor = reduce(add, (penalty(e)>0 for e in upper_z))

        res: Tensor = upper_penalty + lower_penalty
        #count = count_up + count_low
        loss: Tensor = res.sum() / (w * h)**2
        #loss: Tensor = res.sum()
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
        #print(round(loss.item(),1))
        return loss

# w1*CE(source_probs, source_gt) + w2*NaivePenalty(target_probs_size, target_gt_size)


class KLPropInv():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        #print('bounds',torch.round(bounds*10**2)/10**2)
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        #if self.fgt:
        #    two = bounds.shape  # scalar or vector
        #else:
        #    _,_,k,two = bounds.shape
        #assert two == 2

        # est_prop is the proportion estimated by the network
        est_prop: Tensor = self.__fn__(probs,self.power)
        #print('est_prop',torch.round(est_prop*10**2)/10**2)
        # gt_prop is the proportion in the ground truth
        if self.curi:
            bounds = bounds[:,:,0,0] 
            #print(bounds.shape)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            #gt_prop1: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
            #print(gt_prop,gt_prop1)
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        #gt_prop = (gt_prop/(w*h)).type(torch.float32).flatten()

        #value = (value/(w*h)).type(torch.float32).flatten()
        #print(gt_prop.shape)
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        #print(log_est_prop.shape)
        loss = -torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->", [est_prop, log_est_prop])

        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
        #print(loss)
        return loss


class CEProp():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        #print('bounds',torch.round(bounds*10**2)/10**2)
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        #if self.fgt:
        #    two = bounds.shape  # scalar or vector
        #else:
        #    _,_,k,two = bounds.shape
        #assert two == 2

        # est_prop is the proportion estimated by the network
        est_prop: Tensor = self.__fn__(probs,self.power)
        #print('est_prop',torch.round(est_prop*10**2)/10**2)
        # gt_prop is the proportion in the ground truth
        if self.curi:
            bounds = bounds[:,:,0,0] 
            #print(bounds.shape)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            #gt_prop1: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
            #print(gt_prop,gt_prop1)
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        #gt_prop = (gt_prop/(w*h)).type(torch.float32).flatten()

        #value = (value/(w*h)).type(torch.float32).flatten()
        #print(gt_prop.shape)
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        #print(log_est_prop.shape)
        loss = - torch.einsum("bc,bc->", [gt_prop, log_est_prop])

        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
        #print(loss)
        return loss

class CDALoss():
    def __init__(self, **kwargs):
        self.foo = 1
        # self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, source_probs: Tensor, source_lab: Tensor,target_probs: Tensor,target_label: Tensor,bounds: Tensor, w1, w2):
        ce_loss = CrossEntropy(idc=[0,1,2], weights=[1,1,1], dtype='torch.float32')
        np_loss = NaivePenalty(idc=[0,1,2], fn='soft_size')
        loss_1 = w1*ce_loss(source_probs, source_lab,torch.randn(1))
        loss_2 = w2*np_loss(target_probs,target_label,bounds)
        loss = loss_1+loss_2
        return loss, loss_1.item(), loss_2.item()


class DirectLag():
    """
    Implementation in the same fashion as the log-barrier
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        def penalty(z: Tensor) -> Tensor:
            assert z.shape == ()

            return torch.max(torch.zeros_like(z), z)**2

        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        #print(probs.shape,target.shape)
        #assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2
        # assert k == 1  # Keep it simple for now
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]
        #if torch.rand(1).item()>0.999:
        #    print(lower_b, upper_b, value)

        gamma: float = 0.01
        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = (value - upper_b).type(torch.float32).flatten()
        lower_z: Tensor = (lower_b - value).type(torch.float32).flatten()

        g_beta1 = -upper_z
        g_beta2 = lower_z

        #print(g_beta1)
        n_beta1_1 = max(0, gamma * g_beta1[0])
        n_beta1_2 = max(0, gamma * g_beta1[1])
        n_beta2_1 = max(0, gamma * g_beta2[0])
        n_beta2_2 = max(0, gamma * g_beta2[1])


        res: Tensor = g_beta1[0]*n_beta1_1 + g_beta1[1]*n_beta1_2+ g_beta2[0]*n_beta2_1+ g_beta2[1]*n_beta2_2

        #loss: Tensor = res.sum() / (w * h)
        loss: Tensor = res.sum()

        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss



# class NaivePenalty():
#     def __init__(self, **kwargs):
#         self.idc: List[int] = kwargs["idc"]
#         self.C = len(self.idc)
#         self.dtype = kwargs["dtype"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#         self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
#
#     def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)
#         assert probs.shape == target.shape
#
#
#         b, c, w, h = probs.shape  # type: Tuple[int, int, int, int]
#
#         k = bounds.shape[2]  # scalar or vector
#         value: Tensor = self.__fn__(probs[:, self.idc, ...])
#         lower_b = bounds[:, self.idc, :, 0]
#         upper_b = bounds[:, self.idc, :, 1]
#
#         assert value.shape == (b, self.C, k), value.shape
#         assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape
#
#         too_big: Tensor = (value > upper_b).type(self.dtype)
#         too_small: Tensor = (value < lower_b).type(self.dtype)
#
#         big_pen: Tensor = (value - upper_b) ** 2
#         small_pen: Tensor = (value - lower_b) ** 2
#
#         res = too_big * big_pen + too_small * small_pen
#
#         loss: Tensor = res / (w * h)
#
#         return loss.mean()


# class combine_op():
#     def __init__(self, **kwargs):
#         ziped = zip(kwargs["f_args"], kwargs["fns"])
#         self.fns = [eval(fn)(**arg, dtype=kwargs['dtype'], fn=kwargs['fn']) for arg, fn in ziped]
#         self.operator = getattr(__import__("operator"), kwargs["op"])
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def __call__(self, *args, **kwargs):
#         results = map(lambda fn: fn(*args, **kwargs), self.fns)
#         return reduce(self.operator, results)


# class Partial_Size(combine_op):
#     def __init__(self, **kwargs):
#         self.fns = [CrossEntropy(**kwargs['f_args'][0], dtype=kwargs['dtype']),
#                     NaivePenalty(**kwargs['f_args'][1], dtype=kwargs['dtype'], fn="soft_size")]
#         self.operator = getattr(__import__("operator"), "add")


class Pathak(CrossEntropy):
    def __init__(self, **kwargs):
        self.ignore = kwargs["ignore"]
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target) and sset(target, [0, 1])
        assert probs.shape == target.shape

        with torch.no_grad():
            fake_mask: Tensor = torch.zeros_like(probs)
            for i in range(len(probs)):
                fake_mask[i] = self.pathak_generator(probs[i], target[i], bounds[i])
                self.holder_size = fake_mask[i].sum()

        return super().__call__(probs, fake_mask, bounds)

    def pathak_generator(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        _, w, h = probs.shape

        # Replace the probabilities with certainty for the few weak labels that we have
        weak_labels = target[...]
        weak_labels[self.ignore, ...] = 0
        assert not simplex(weak_labels) and simplex(target)
        lower, upper = bounds[-1]

        labeled_pixels = weak_labels.any(axis=0)
        assert w * h == (labeled_pixels.sum() + (~labeled_pixels).sum())  # make sure all pixels are covered
        scribbled_probs = weak_labels + torch.einsum("cwh,wh->cwh", probs, ~labeled_pixels)
        assert simplex(scribbled_probs)

        u: Tensor
        max_iter: int = 100
        lr: float = 0.00005
        b: Tensor = torch.Tensor([-lower, upper])
        beta: Tensor = torch.zeros(2, torch.float32)
        f: Tensor = torch.zeros(2, *probs.shape)
        f[0, ...] = -1
        f[1, ...] = 1

        for i in range(max_iter):
            exped = - torch.einsum("i,icwh->cwh", beta, f).exp()
            u_star = torch.einsum('cwh,cwh->cwh', probs, exped)
            u_star /= u_star.sum(axis=0)
            assert simplex(u_star)

            d_beta = torch.einsum("cwh,icwh->i", u_star, f) - b
            n_beta = torch.max(torch.zeros_like(beta), beta + lr * d_beta)

            u = u_star
            beta = n_beta

        return probs2one_hot(u)


class Adversarial():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["imtype"]


class BCELoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.dtype = kwargs["dtype"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, d_out: Tensor, label: float):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(d_out,Tensor(d_out.data.size()).fill_(label).to(d_out.device))
        return loss



class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss

def d_loss_calc(pred, label):
    loss_params = {'idc' : [0, 1]}
    criterion = BCELoss(**loss_params, dtype="torch.float32")
    return criterion(pred, label)
