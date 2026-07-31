import torch
import torch.distributed as dist

# copied from https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



def split_heads(matrix, num_heads, head_axis):
    """
    View a 2D attention projection as num_heads independent blocks, so that a batched Newton-Schulz
    orthogonalizes each head on its own instead of coupling every head through one polar factor.
    head_axis is "row" when out_features is num_heads * head_dim, as for the query/key/value
    projections, and "col" when in_features is, as for the output projection.
    """
    assert matrix.ndim == 2, "per-head splitting expects a 2D attention projection"
    rows, cols = matrix.shape
    if head_axis == "row":
        assert rows % num_heads == 0, f"out_features {rows} is not divisible by num_heads {num_heads}"
        return matrix.unflatten(0, (num_heads, -1))

    assert head_axis == "col", f"head_axis must be 'row' or 'col', got {head_axis!r}"
    assert cols % num_heads == 0, f"in_features {cols} is not divisible by num_heads {num_heads}"
    return matrix.unflatten(1, (num_heads, -1)).transpose(0, 1)


def merge_heads(blocks, head_axis):
    if head_axis == "row":
        return blocks.flatten(0, 1)

    return blocks.transpose(0, 1).flatten(1, 2)


def second_momentum_buffer(param, num_heads, head_axis):
    if num_heads is None:
        return torch.zeros_like(param[..., 0:1])

    return param.new_zeros(split_heads(param, num_heads, head_axis)[..., 0:1].shape)


def head_config(group):
    # load_state_dict overwrites a group with the saved one, which for a checkpoint predating
    # per-head Muon carries no head keys at all.
    return group.get("num_heads"), group.get("head_axis", "row")


def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True,
                   num_heads=None, head_axis="row"):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        assert num_heads is None, "per-head splitting does not apply to conv filters"
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    if num_heads is not None:
        update = split_heads(update, num_heads, head_axis)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)
    ################ NorMuon added ###################
    # With the heads split, dim=(-2,-1) spans one head, so the norm restoration below is per-head.
    vnorm = update.norm(dim=(-2,-1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
    ##################################################
    # Shape of whatever was orthogonalized, which with the heads split is one head's block. That
    # keeps a row-split update at the same Frobenius norm as a fused one, so lr carries over.
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    if num_heads is not None:
        update = merge_heads(update, head_axis)
    return update


# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    had_grad = p.grad is not None
                    if not had_grad:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95, num_heads=None, head_axis="row"):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2, num_heads=num_heads,
                        head_axis=head_axis)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            num_heads, head_axis = head_config(group)
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = second_momentum_buffer(p, num_heads, head_axis)
                update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                        beta=group["momentum"], beta2=group["beta2"],
                                        num_heads=num_heads, head_axis=head_axis)
                if group["weight_decay"] and had_grad:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class NorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed counterpart to NorMuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["num_heads"] = group.get("num_heads", None)
                group["head_axis"] = group.get("head_axis", "row")
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon",
                                             "num_heads", "head_axis"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                num_heads, head_axis = head_config(group)
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = second_momentum_buffer(p, num_heads, head_axis)
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                            beta=group["momentum"], beta2=group["beta2"],
                                            num_heads=num_heads, head_axis=head_axis)
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
