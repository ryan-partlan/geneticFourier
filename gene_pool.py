from cand import *


class GenePool:
    def __init__(self, objective, pop_size, num_cos, num_sin, pollirate=0.01, mute_rate=0.01):
        self.objective = objective
        self.pop_size = pop_size
        self.mute_rate = mute_rate
        self.pollirate = pollirate
        self.num_cos = num_cos
        self.num_sin = num_sin
        self.pop = self.let_there_be()

    def let_there_be(self):
        pop = []
        for _ in range(self.pop_size):
            cand = Candidate(len(self.objective))
            cand.init_genotype(self.num_sin, self.num_cos)
            pop.append(cand)
        return pop

    def assign_fitness(self):
        fits = []
        for cand in self.pop:
            cand_fit = np.mean([abs(cand.eval_at(i) - x) for i, x in enumerate(self.objective)])
            fits.append(cand_fit)
            cand.fit = cand_fit
        return fits

    def sort_pop(self):
        self.pop = sorted(self.pop, key=lambda x: x.fit)

    def mutate(self):
        for cand in self.pop:
            if random.random() < self.mute_rate:
                cand.mutate()
            if random.random() < self.mute_rate:
                cand.mute_first_coef()

    def cross_pars(self, openings):
        breed_pairs = [pair for pair in permutations(self.pop, 2) if pair[0] != pair[1]]
        random.shuffle(breed_pairs)
        for cand1, cand2 in breed_pairs[:openings]:
            new_cand = Candidate(len(self.objective))
            if random.randint(0, 1) == 1:
                new_cand.init_from_parents(cand1, cand2)
            else:
                new_cand.init_crossbreed(cand1, cand2)
            self.pop.append(new_cand)

    def cross_pollinate(self):
        breed_pairs = [pair for pair in combinations(self.pop, 2) if pair[0] != pair[1]]
        random.shuffle(breed_pairs)
        for cand1, cand2 in breed_pairs:
            if random.random() < self.pollirate:
                i = random.randint(0, min(cand1.num_cos, cand2.num_cos) - 1)
                cache = cand1.cos_coeffs[i]
                cand1.cos_coeffs[i] = cand2.cos_coeffs[i]
                cand2.cos_coeffs[i] = cache
                j = random.randint(0, min(cand1.num_sins, cand2.num_sins) - 1)
                cache = cand1.sin_coeffs[j]
                cand1.sin_coeffs[j] = cand2.sin_coeffs[j]
                cand2.sin_coeffs[j] = cache


    def elitism(self, elite_rate=0.1):
        self.sort_pop()
        old_size = len(self.pop)
        new_size = int(old_size * (1 - elite_rate))
        self.pop = self.pop[:new_size]
        return old_size - new_size

    def grow_appendages(self):
        for cand in self.pop:
            if random.randint(0, 1) == 1:
                cand.add_cos_coef()
            else:
                cand.add_sin_coef()

    def evolve(self, epsilon=0.001, num_gens=10, lookback=10):
        avg_fits = []
        for i in range(num_gens):
            print(f"Generation: {i}")
            fits = self.assign_fitness()
            self.sort_pop()
            avg_fit = np.mean(fits)
            # print(len(self.pop))
            openings = self.elitism()
            self.cross_pars(openings)
            self.cross_pollinate()
            avg_fits.append(avg_fit)
            if i > lookback and np.var(avg_fits[-1 * lookback:]) < epsilon:
                self.grow_appendages() # Randomly add either a cos or sin coef to every candidate.
            print("avg", avg_fit, "min", min(fits))
        self.sort_pop()
        return self.pop[0]

if __name__ == "__main__":
    x_axis = range(1, 100)
    obj = np.sin(x_axis)
    # obj = np.log(x_axis)
    # obj = [x**4 for x in x_axis]
    gp = GenePool(obj, 100, 1, 1, mute_rate=0.01, pollirate=0.01)
    ngens = 400
    best_cand = gp.evolve(num_gens=ngens, epsilon=0.001, lookback=30)
    print([best_cand.eval_at(x) for x in x_axis])
    print(best_cand.cos_coeffs)
    print(best_cand.sin_coeffs)
    plt.title(str(ngens))
    plt.plot(x_axis, obj)
    plt.plot(x_axis, [best_cand.eval_at(x) for x in x_axis])
    plt.show()



