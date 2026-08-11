import torch
import torch.distributed as dist

# Polar Express (https://arxiv.org/abs/2505.16932): per-step coefficients for the same odd quintic
# the Muon iteration uses. Early steps are aggressive to lift small singular values; the schedule
# anneals to (1.875, -1.25, 0.375), the order-5 Newton-Schulz for the matrix sign, whose p(1) = 1
# makes 1 a fixed point. Muon's single (3.4445, -4.7750, 2.0315) has p(1) = 0.701, so 1 is NOT a
# fixed point: it parks the spectrum in a band around 1 and more steps only reshuffle it there.
ABC_LIST = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# Safety factor for numerical stability, excluding the last polynomial (already the exact iteration).
ABC_LIST_STABLE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


def zeropower_via_polar_express(G, steps=8):
    """Orthogonalization via the Polar Express schedule: same contract, shape handling and per-step
    cost (3 matmuls) as zeropower_via_newtonschulz5, but it converges.

    8 steps rather than the quintic's 5, which costs 24 matmuls against 15. Five would already fix
    most of what the split below depends on -- against the row-split-leaves-the-norm-unchanged
    identity the quintic runs 16-18% off and 5 steps land within 0.9% -- but it stops well short of
    convergence, at s_min 0.81 and mean |s-1| 0.083 per block. 8 reaches 0.997 and 0.002, and that
    converged regime is where the published Muon result for this schedule was measured.
    """
    assert G.ndim >= 2
    should_transpose = G.size(-2) > G.size(-1)

    X = G.bfloat16()
    if should_transpose:
        X = X.mT

    # The 1.01 margin keeps the steep leading coefficient from overshooting on the first step.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[min(step, len(ABC_LIST_STABLE) - 1)]
        S = X @ X.mT
        # a*X + b*S@X + c*S@S@X, grouped as (a*I + (b*I + c*S) @ S) @ X to save a matmul.
        Y = c * S
        Y.diagonal(dim1=-2, dim2=-1).add_(b)
        Y = Y @ S
        Y.diagonal(dim1=-2, dim2=-1).add_(a)
        X = Y @ X

    if should_transpose:
        X = X.mT
    return torch.nan_to_num(X)


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


def hyperball_step(p, update, lr, state, num_heads=None, head_axis="row"):
    """Step along the update, then project the weight back onto the sphere it started on.

    Hyperball, from "Fantastic Pretraining Optimizers and Where to Find Them II" (arXiv 2606.16899):
    W <- R * Normalize(W - lr * R * Normalize(u)), Frobenius throughout, R fixed for the run at the
    initial weight norm. Weight and step then share units, so lr is the fraction of the radius
    travelled per step rather than a step in whatever scale the base update happens to carry.

    The sphere is per head wherever the update is, one radius per block rather than one for the
    matrix. A single radius pins only the sum of squares, which lets one head grow at another's
    expense -- exactly the coupling between heads that splitting them exists to remove. Per block,
    NorMuon's norm restoration no longer sets the relative size of the heads, since each is pinned to
    its own radius; what survives of it is the direction its per-row rescaling picks inside a block.

    Weight decay is dropped rather than ignored: scaling W by c before a step of fixed norm and then
    renormalizing lands exactly where stepping with lr / c does, so under the projection decay is
    only a learning rate in disguise.

    The unit is always the one normuon_update orthogonalized, which for a parameter of more than two
    dimensions is not the whole tensor: a 3D expert bank is a BATCH of independent matrices there, so
    it gets a radius each, while a 4D conv filter is a single matrix flattened to (out, in*k). One
    radius over an expert bank would pin only the sum of squares across its experts, which is the
    coupling the per-head split exists to remove, one axis over. A caller with no orthogonalized unit
    to follow -- the Adam branch -- passes a flat view instead, one sphere per parameter.

    A parameter whose norm is 0 has no sphere: every direction is the same point, and the projection
    would pin it there for the whole run through a divide that never errors. Zero-initialized
    parameters are common (a zero-centered norm's gain, any residual-branch output init), so this
    refuses rather than freezing them silently.
    """
    if num_heads is None:
        if p.ndim == 4:
            assert p.is_contiguous(), "projecting a conv filter in place needs a viewable parameter"
            update = update.reshape(p.size(0), -1)
            p = p.view(p.size(0), -1)
        dims = (0,) if p.ndim == 1 else (-2, -1)
    else:
        p, update, dims = split_heads(p, num_heads, head_axis), split_heads(update, num_heads, head_axis), (-2, -1)

    def norm(x):
        return x.norm(dim=dims, keepdim=True, dtype=torch.float32).add(1e-10)

    radius = state.get("radius")
    if radius is None:
        radius = state["radius"] = p.norm(dim=dims, keepdim=True, dtype=torch.float32)
        assert radius.gt(0).all(), (
            "hyperball needs a nonzero weight norm to define a sphere; a zero-initialized parameter "
            f"would be pinned at zero for the whole run (shape {tuple(p.shape)})"
        )
    p.sub_(update * (lr * radius / norm(update)))
    p.mul_(radius / norm(p))


def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=8, nesterov=True,
                   num_heads=None, head_axis="row"):
    """8 Polar Express steps replace 5 of the fixed-coefficient quintic. The quintic's spread is not
    a step-count problem -- p(1) = 0.701 for its coefficients, so 1 is not a fixed point and extra
    steps only reshuffle the spectrum inside a band -- which is why the swap is to a schedule that
    converges rather than to more of the same iteration.

    The exactness is load-bearing for the split below and for any caller scaling by the block shape:
    with the quintic a row split misses the fused update norm by 16-18% and a 0.2 RMS target lands
    7-11% out, differing in sign across shapes so groups drift apart relative to each other. Both
    fall to 0.1% here.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        assert num_heads is None, "per-head splitting does not apply to conv filters"
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    if num_heads is not None:
        update = split_heads(update, num_heads, head_axis)
    update = zeropower_via_polar_express(update, steps=ns_steps)
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


def fp32_accumulator(state, p):
    """The fp32 shadow an update is applied to, for a parameter held in lower precision.

    bf16 keeps 7 mantissa bits, so a parameter at 0.125 has a ULP of 9.8e-4 and rounds any update
    below half of that straight back to itself. Under a 4e-4 Adam step every zero-centered RMSNorm
    gain in a bf16 model therefore climbs to exactly 0.125 and stops, and parameters at 1.0 -- a
    gated norm's init, dt_bias -- never move at all: measured across four runs, whose dt_bias was
    bit-identical at 1.00 under different seeds. Weight decay is lost the same way, being smaller
    still. Accumulating in fp32 and rounding into the parameter is what lets increments below an
    ULP add up to one, at 4 bytes per parameter. Kahan summation (arXiv 2010.06192, and torchdistx's
    AnyPrecisionAdamW) buys the same thing for 2, and measures at least as well; it is the move if
    the 4 bytes ever bind, which on one 96GB card they do not.

    Returns p itself when p is already fp32, so those parameters keep their exact prior behavior.
    The other optimizers here share the flaw and are deliberately left alone: this is the one that
    is exercised, and an untested change to a code path is worse than a known one.

    Momentum buffers stay in the parameter's dtype on purpose. A decaying average of gradients is
    transient and tracks the gradient's own scale, unlike a parameter, which is a running sum where
    a dropped increment never comes back. Holding momentum in fp32 would cost as much as the model.
    """
    if p.dtype == torch.float32:
        return p
    master = state.get("master")
    if master is None:
        master = state["master"] = p.detach().float().clone()
    return master


def restore_fp32_state(optimizer, state_dict):
    """Undo Optimizer.load_state_dict's cast of float state to the parameter dtype.

    That cast (_process_value_according_to_param_policy) assumes every floating-point state tensor
    matches its parameter's dtype, which for a bf16 parameter downcasts the accumulator above and
    silently reinstates the stall it exists to prevent. The hyperball radius needs the same
    exemption: rounded to bf16 it moves the sphere by up to 0.4% on resume, which is a weight rescale
    the run never asked for. The id -> parameter mapping is built the way load_state_dict builds its
    own.
    """
    saved_ids = [pid for group in state_dict["param_groups"] for pid in group["params"]]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    id_map = dict(zip(saved_ids, params))
    for pid, saved in state_dict["state"].items():
        for key in ("master", "radius"):
            value = saved.get(key)
            if value is not None:
                p = id_map[pid]
                optimizer.state[p][key] = value.detach().clone().to(device=p.device)


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
                # Opt-in, because it redefines lr for the group and not every parameter can take it:
                # a zero-initialized one has no sphere at all (hyperball_step refuses), and the rate
                # matching an Adam step differs by orders of magnitude between a matrix and a short
                # 1-D parameter, so one group cannot hold both.
                group["hyperball"] = group.get("hyperball", False)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon",
                                             "hyperball"}
                # Under the projection a decay is only a learning rate in disguise, so a value here
                # would silently do nothing rather than what it says.
                assert not (group["hyperball"] and group["weight_decay"]), \
                    "a hyperball group cannot carry weight decay; the projection absorbs it into lr"
        super().__init__(param_groups, dict())

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        restore_fp32_state(self, state_dict)

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
                    target = fp32_accumulator(state, p)
                    hyperball_step(target, update.reshape(p.shape), group["lr"], state, num_heads, head_axis)
                    if target is not p:
                        p.copy_(target)
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
                    target = fp32_accumulator(state, p)
                    if group["hyperball"]:
                        # One sphere per parameter, via a flat view: nothing orthogonalized these, so
                        # there is no block structure for a radius to follow. lr is a radius fraction
                        # here too, so it is NOT the additive branch's rate.
                        hyperball_step(target.view(-1), update.reshape(-1), group["lr"], state)
                    else:
                        if group["weight_decay"] and had_grad:
                            target.mul_(1 - group["lr"] * group["weight_decay"])
                        target.add_(update, alpha=-group["lr"])
                    if target is not p:
                        p.copy_(target)

        return loss
