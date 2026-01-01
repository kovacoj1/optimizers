import torch
from torch.autograd import grad


class ExtendedKalmanFilter(torch.optim.Optimizer):
    # only one parameter group can be used!
    # add @property q(t)
    def __init__(self, params, eta = 1e3, eps = 1e-3, q = 1e-6, tau = 1):
        self.eta, self.eps, self.q, self.tau = eta, eps, q, tau

        defaults = dict(
                    eta = self.eta,
                    eps = self.eps,
                    q = self.q,
                    tau = self.tau
                )
        
        super(ExtendedKalmanFilter, self).__init__(params, defaults)

        self.numel = sum(param.numel() for group in self.param_groups for param in group['params'] if param.requires_grad)

        self.P = torch.eye(self.numel) / self.eta # error covariance matrix
        self.Q = self.q * torch.eye(self.numel) # artificial process noise

    def jacobian(self, targets):
        # needs to be tested (nn design)
        J = torch.empty(targets.shape[0], self.numel)
        
        for i in range(targets.shape[0]):
            J[i] = torch.hstack([d.view(1, -1) if d is not None else torch.tensor([0.]).view(1, -1) for d in grad(targets[i], self.param_groups[0]['params'], create_graph=True, retain_graph=True, allow_unused=True)])

        return J
    
    def loss(self, errors):
        # MSE loss, not divided by number of data, doesn't matter
        return errors.T @ errors
    
    @torch.no_grad()
    def update_weights(self, updates):
        # maybe could be optimzed using torch.chunk
        # Add the updates into the model
        start_idx = 0
        for group in self.param_groups:
            for param in group['params']:
                param.data.add_(updates[start_idx:start_idx + param.numel()].view(param.size()))
                start_idx += param.numel()

    def step(self, closure = None):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # errors need to be computed from closure
        # closure (callable) - reevaluates the model and returns the loss, in our case the errors
        errors = closure()
        
        H = self.jacobian(errors)

        A = torch.inverse(torch.eye(errors.shape[0]) / self.eta + H @ self.P @ H.T)

        K = self.P @ H.T @ A

        updates = -(K @ errors).view(-1)

        self.update_weights(updates)

        self.P -= K @ H @ self.P + self.Q

        return self.loss(closure())


# helpful code
def jacobian(params, target):
    numel = sum(p.numel() for p in params if p.requires_grad)
    
    J = torch.empty(target.shape[0], numel)
    for i in range(target.shape[0]):
        J[i] = torch.hstack([d.view(1, -1) if d is not None else torch.tensor([0.]).view(1, -1) for d in grad(target[i], params, create_graph=True, retain_graph=True, allow_unused=True)])

    return J

def update_weights(param_groups, updates):
    updates = updates.view(-1)

    start_idx = 0
    for group in param_groups:
        for param in group['params']:
            param.data.add_(updates[start_idx:start_idx + param.numel()].view(param.size()))
            start_idx += param.numel()
