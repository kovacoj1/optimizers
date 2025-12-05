import torch

class Annealing(torch.optim.Optimizer):
    def __init__(self, params, defaults={'alpha': 0.999, 'helper_lr': 0.1, 'steps': 100, 'initial_T': 10.0}):
        super().__init__(params, defaults)
        self.numel = sum(p.numel() for group in self.param_groups for p in group['params'])
        self.steps = defaults['steps']
        # Dummy optimizer for scheduling temperature (treated as 'lr')
        self.temperature = defaults['initial_T']
        dummy_param = torch.tensor(self.temperature)
        self.dummy_opt = torch.optim.SGD([dummy_param], lr=self.temperature)
        self.temperature_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.dummy_opt, gamma=defaults['alpha'])
        # Helper Adam with custom lr
        self.helper = torch.optim.Adam(self.param_groups, lr=defaults['helper_lr'])

    @property
    def params(self):
        return torch.cat([p.flatten() for group in self.param_groups for p in group['params']])

    @torch.no_grad()
    def update_weights(self, update):
        update = update.flatten()
        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()
                param.data = update[offset: offset + numel].view_as(param)
                offset += numel

    @torch.no_grad()
    def mutate(self):
        return self.params + torch.randn_like(self.params) * self.temperature

    def directional(self, closure):
        self.helper.state.clear()
        for _ in range(self.steps):
            self.helper.zero_grad()
            loss = closure()
            loss.backward()
            self.helper.step()
        return closure()

    def step(self, closure):
        variants = [self.params.clone(), self.mutate().clone()]
        Fs = torch.zeros(2)
        for idx in range(2):
            self.update_weights(variants[idx])
            Fs[idx] = self.directional(closure).item()
            variants[idx] = self.params.clone()  # Capture refined params

        delta = Fs[1] - Fs[0]
        prob = torch.exp(-delta / self.temperature)
        idx = 1 if (delta <= 0 or torch.rand(1) < prob) else 0  # Always accept better; prob for worse

        self.update_weights(variants[idx])
        # Update temperature via scheduler
        self.temperature_scheduler.step()
        self.temperature = self.temperature_scheduler.get_last_lr()[0]
        return Fs[idx]
