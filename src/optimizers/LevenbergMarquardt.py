import torch
from torch.autograd import grad
    

class LevenbergMarquardt(torch.optim.Optimizer):
    # only one parameter group can be used!
    # need to add comments
    # update defaults
    # add loss history; batches can be controled from closure()
    def __init__(self, params, mu = 10**3, mu_factor = 5, m_max = 10):
        self.mu = mu
        self.mu_factor = mu_factor
        self.m_max = m_max

        defaults = dict(mu = self.mu,
                        mu_factor = self.mu_factor,
                        m_max = self.m_max
                    )
        
        super(LevenbergMarquardt, self).__init__(params, defaults)

        self.numel = sum(param.numel() for group in self.param_groups for param in group['params'] if param.requires_grad)
        # self.numel = reduce(lambda total, p: total + p.numel(), self.param_groups, 0)

    # @torch.compile ?
    def jacobian(self, targets):
        # needs to be tested (nn design)
        J = torch.empty(targets.shape[0], self.numel)
        
        for i in range(targets.shape[0]):
            J[i] = torch.hstack([d.view(1, -1) if d is not None else torch.tensor([0.]).view(1, -1) for d in grad(targets[i], self.param_groups[0]['params'], create_graph=True, retain_graph=True, allow_unused=True)])

        return J
    
    @torch.no_grad()
    def loss(self, errors):
        # MSE loss, not divided by number of data, doesn't matter
        return errors.T @ errors
    
    @torch.no_grad()
    def update_weights(self, update):
        update = update.view(-1)

        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()

                param.add_(update[offset: offset + numel].view_as(param))
                offset += numel

    def step(self, closure = None):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # errors need to be computed from closure
        # closure (callable) - reevaluates the model and returns the loss, in our case the errors
        errors = closure()
        
        # compute Jacobian matrix
        J = self.jacobian(errors)

        # compute updates %torch.diag(J.T @ J)%
        updates = -torch.inverse(J.T @ J + (self.mu+1e-8)*torch.eye(self.numel)) @ J.T @ errors

        self.update_weights(updates)

        # line search for mu
        for m in range(self.m_max):

            # check if loss has decreased
            if self.loss(closure()) < self.loss(errors):
                break

            # restore weights
            self.update_weights(update = -updates)

            self.mu *= self.mu_factor

            # compute new updates
            updates = -torch.inverse(J.T @ J + (self.mu+1e-8)*torch.eye(self.numel)) @ J.T @ errors

            # update weights
            self.update_weights(update = +updates)

        if m < self.m_max:
            self.mu /= self.mu_factor

        # how to return break?, should I return loss?
        # -> returning loss, mu can be controled by
        #  optimizer.mu before running optimizer.step(...)
        return self.loss(closure()).item()

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
