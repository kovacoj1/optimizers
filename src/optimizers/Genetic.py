import torch
from .Newton import Newton


class Genetic(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {}) 

        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.elite_ratio = 0.2

        self.best_genome = self.params
        self.best_fitness = float('inf')

        self.numel = sum(
            p.numel() for group in self.param_groups for p in group['params']
        )
        self.pop_size = 100 # max(int(self.numel**0.5), 10)

        self.population = self.params.unsqueeze(0).repeat(self.pop_size, 1)
        self.population += torch.randn_like(self.population) * 1e-0

        self.helper = torch.optim.Adam(self.param_groups)
        # self.helper = Newton(self.param_groups[0]['params'])

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
    def mutate(self, genome):
        mask = torch.rand_like(genome) < self.mutation_rate
        noise = torch.randn_like(genome)*self.mutation_strength**2

        genome[mask] += noise[mask]

        return genome

    @torch.no_grad
    def crossover(self, parent1, parent2):
        crossover_point = torch.randint(1, self.numel, (1,))[0]

        child1 = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = torch.cat((parent2[:crossover_point], parent1[crossover_point:]))

        return child1.clone(), child2.clone()

    # @torch.no_grad
    def directional(self, closure, steps=0):
        self.helper.state.clear()

        for _ in range(steps):
            self.helper.zero_grad()
            closure().backward()
            self.helper.step()
            # self.helper.step(closure)

        return closure()

    # @torch.no_grad
    def step(self, closure):
        loss = torch.empty(self.pop_size)

        for idx in range(self.pop_size):            
            self.update_weights(self.population[idx])
            loss[idx] = self.directional(closure)
            self.population[idx] = self.params.clone()

        # TRACK BEST INDIVIDUAL (ELITISM)
        best_idx = torch.argmin(loss)
        current_best_fitness = loss[best_idx].item()

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_genome = self.population[best_idx].clone()

        # SELECTION (Truncation Selection)
        num_parents = max(1, int(self.pop_size * self.elite_ratio))
        sorted_indices = torch.argsort(loss, descending=False)
        parent_indices = sorted_indices[:num_parents]
        parents = self.population[parent_indices]

        # REPRODUCTION
        new_population = [self.best_genome.clone()]

        while len(new_population) < self.pop_size:
            p1_idx, p2_idx = torch.randint(0, num_parents, (2,)).unbind()
            
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]

            # Crossover
            child1, child2 = self.crossover(p1, p2)

            # Mutation
            new_population.append(self.mutate(child1))
            if len(new_population) < self.pop_size:
                new_population.append(self.mutate(child2))

        self.population = torch.stack(new_population)

        # pdate best weights for external calls to model(x)
        self.update_weights(self.best_genome)

        return closure()
