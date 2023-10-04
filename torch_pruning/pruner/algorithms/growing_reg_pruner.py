import torch
import torch.nn as nn
import typing 
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ... import ops

class GrowingRegPruner(MetaPruner):
    """ pruning with growing regularization
    https://arxiv.org/abs/2012.09243
    """
    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-5, # regularization coefficient
        delta_reg=1e-5, # increment of regularization coefficient
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        ch_sparsity: float = 0.5,  # channel/dim sparsity, also known as pruning ratio
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, # layer-specific sparsity, will cover ch_sparsity if specified
        max_ch_sparsity: float = 1.0, # maximum sparsity. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
    ):
        super(GrowingRegPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            max_ch_sparsity=max_ch_sparsity,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
        )
        self.base_reg = reg
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

        group_reg = {}
        for group in self._groups:
            group_reg[group] = torch.ones(len(group[0].idxs)) * self.base_reg
        self.group_reg = group_reg
        self.delta_reg = delta_reg

    def update_reg(self):
        for group in self._groups:
            group_l2norm_sq = self.estimate_importance(group)
            if group_l2norm_sq is None:
                continue
            reg = self.group_reg[group]
            standarized_imp = (group_l2norm_sq.max() - group_l2norm_sq) / \
                (group_l2norm_sq.max() - group_l2norm_sq.min() + 1e-8)  # => [0, 1]
            reg = reg + self.delta_reg * standarized_imp.to(reg.device)
            self.group_reg[group] = reg

    def step(self, interactive=False): 
        super(GrowingRegPruner, self).step(interactive=interactive)
        # update the group list after pruning
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        group_reg = {}
        for group in self._groups:
            group_reg[group] = torch.ones(len(group[0].idxs)) * self.base_reg
        self.group_reg = group_reg

    def regularize(self, model, bias=False):
        for i, group in enumerate(self._groups):
            group_l2norm_sq = self.estimate_importance(group)
            if group_l2norm_sq is None:
                continue
            gamma = self.group_reg[group]
            for k, (dep, idxs) in enumerate(group):
                layer = dep.layer
                pruning_fn = dep.pruning_fn

                if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine == True and layer not in self.ignored_layers:
                    if layer.weight.grad is None: continue

                    root_idxs = group[k].root_idxs
                    _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))
                    
                    layer.weight.grad.data[idxs].add_(_gamma.to(layer.weight.device) * layer.weight.data[idxs])
                    if bias and layer.bias is not None:
                        layer.bias.grad.data[idxs].add_(_gamma.to(layer.weight.device) * layer.bias.data[idxs])
                elif isinstance(layer, (nn.modules.conv._ConvNd, nn.Linear)):
                    if pruning_fn in [function.prune_conv_out_channels, function.prune_linear_out_channels] and layer not in self.ignored_layers:
                        if layer.weight.grad is None: continue

                        root_idxs = group[k].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                        w = layer.weight.data[idxs]
                        g = w * _gamma.to(layer.weight.device).view(-1, *([1]*(len(w.shape)-1)))
                        layer.weight.grad.data[idxs] += g

                        if bias and layer.bias is not None:
                            b = layer.bias.data[idxs]
                            g = b * _gamma.to(layer.weight.device)
                            layer.bias.grad.data[idxs] += g
                    elif pruning_fn in [function.prune_conv_in_channels, function.prune_linear_in_channels]:
                        if layer.weight.grad is None: continue
                        root_idxs = group[k].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                        w = layer.weight.data[:, idxs]
                        g = w * _gamma.to(layer.weight.device).view(1, -1, *([1]*(len(w.shape)-2)))
                        layer.weight.grad.data[:, idxs] += g
