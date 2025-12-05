import torch
from torch.autograd import grad


class Newton(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

        self.numel = sum(
            p.numel() for group in self.param_groups for p in group['params']
        )
    
    @property
    def params(self):
        return torch.cat([
            p.flatten() for group in self.param_groups for p in group['params']
        ])

    @torch.no_grad()
    def update_weights(self, update):
        update = update.flatten()

        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()

                param.add_(update[offset: offset + numel].view_as(param))
                offset += numel

    def step(self, closure: callable):
        grads = grad(closure(), self.param_groups[0]['params'], create_graph=True)

        g = torch.cat([g.reshape(-1) for g in grads])
        H = torch.empty(self.numel, self.numel)

        for idx in range(g.shape[0]):
            H[idx] = torch.hstack([
                d.view(1, -1) for d in grad(g[idx], self.param_groups[0]['params'], create_graph=True, retain_graph=True)
            ])
        H += 1e-4 * torch.eye(self.numel) # damping for num. stability

        self.update_weights(
            torch.linalg.solve(H, -g)
        )
