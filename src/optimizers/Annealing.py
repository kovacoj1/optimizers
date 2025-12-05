import torch


class Annealing(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {}) 

        self.numel = sum(
            p.numel() for group in self.param_groups for p in group['params']
        )
        self.temperature = 1

    @property
    def params(self):
        return torch.cat([
            p.flatten() for group in self.param_groups for p in group['params']
        ])

    @torch.no_grad
    def update_weights(self, update):
        update = update.flatten()

        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()

                param.data = update[offset: offset + numel].view_as(param)
                offset += numel

    @torch.no_grad
    def mutate(self):
        return self.params + torch.randn_like(
            self.params
        )*self.temperature

    @torch.no_grad
    def step(self, closure):
        variants = [self.params, self.mutate().clone()]
        Fs = torch.empty(2)

        for idx, params in enumerate(variants):
            self.update_weights(params)
            Fs[idx] = closure().item()

        idx = torch.rand(1) < torch.exp(
            (Fs[0] - Fs[1]) / self.temperature
        )

        self.update_weights(variants[idx])
        self.temperature *= 1 - 1e-3

        return closure()
