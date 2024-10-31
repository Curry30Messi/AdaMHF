import torch
from collections import defaultdict
import math
from torch.optim.optimizer import Optimizer


def define_optimizer(config, model):
    optimizer_mapping = {
        'SGD': torch.optim.SGD,
        'AdamW': torch.optim.AdamW,
        'Adam': torch.optim.Adam,
        'RAdam': RAdam,
        'PlainRAdam': PlainRAdam,
        'Lookahead': lambda params, lr, weight_decay: Lookahead(
            torch.optim.Adam(params, lr=lr, weight_decay=weight_decay))
    }

    if config.optimizer in optimizer_mapping:
        optimizer = optimizer_mapping[config.optimizer](
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer [{config.optimizer}] is not implemented")

    return optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                param_data = param.data.float()
                state = self.state[param]

                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param_data)
                    state['exp_avg_sq'] = torch.zeros_like(param_data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(param_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(param_data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.buffer[state['step'] % 10]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    param_data.add_(param_data, alpha=-group['weight_decay'] * group['lr'])

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    param_data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    param_data.add_(exp_avg, alpha=-step_size)

                param.data.copy_(param_data)

        return loss


class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                param_data = param.data.float()
                state = self.state[param]

                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param_data)
                    state['exp_avg_sq'] = torch.zeros_like(param_data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(param_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(param_data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    param_data.add_(param_data, alpha=-group['weight_decay'] * group['lr'])

                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)
                    ) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    param_data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    param_data.add_(exp_avg, alpha=-step_size)

                param.data.copy_(param_data)

        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_param in group['params']:
            if fast_param.grad is None:
                continue
            param_state = self.state[fast_param]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_param.data)
                param_state['slow_buffer'].copy_(fast_param.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_param.data - slow)
            fast_param.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(key) if isinstance(key, torch.Tensor) else key): value
            for key, value in self.state.items()
        }
        return {
            'state': fast_state_dict['state'],
            'slow_state': slow_state,
            'param_groups': fast_state_dict['param_groups'],
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
        super().load_state_dict({
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        })
        self.param_groups = self.base_optimizer.param_groups
        for name, default in self.defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)