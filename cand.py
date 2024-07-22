import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations


class Candidate:
    def __init__(self, x_range, num_sins=0, num_cos=0):
        self.x_range = x_range / 2
        self.first_coef = 0
        self.num_sins = 0
        self.num_cos = 0
        self.cos_coeffs = []
        self.sin_coeffs = []
        self.fit = 0

    def init_genotype(self, num_sins, num_cos):
        for i in range(num_sins):
            self.add_sin_coef()
        for j in range(num_cos):
            self.add_cos_coef()
        self.first_coef = random.uniform(-1, 1)

    def eval_at(self, x):
        res = self.first_coef
        for i, cos_coef in enumerate(self.cos_coeffs):
            in_term = ((i + 1) * np.pi * x) / self.x_range
            res += cos_coef * np.cos(in_term)
        for j, sin_coef in enumerate(self.sin_coeffs):
            in_term = ((j + 1) * np.pi * x) / self.x_range
            res += sin_coef * np.sin(in_term)
        return res

    def add_cos_coef(self, val=None):
        if val:
            self.cos_coeffs.append(val)
        else:
            self.cos_coeffs.append(random.uniform(-1, 1))
        self.num_cos += 1

    def mutate(self):
        cos_targ = random.randint(0, len(self.cos_coeffs))
        self.cos_coeffs[cos_targ] += random.uniform(-1, 1)
        sin_targ = random.randint(0, len(self.sin_coeffs))
        self.sin_coeffs[sin_targ] += random.uniform(-1, 1)

    def add_sin_coef(self, val=None):
        if val:
            self.sin_coeffs.append(val)
        else:
            self.sin_coeffs.append(random.uniform(-1, 1))
        self.num_sins += 1

    def init_from_parents(self, cand1, cand2):
        self.sin_coeffs = cand1.sin_coeffs.copy()
        self.num_sins = len(self.sin_coeffs)
        self.cos_coeffs = cand2.cos_coeffs.copy()
        self.num_cos = len(self.cos_coeffs)
        self.first_coef = cand2.first_coef

    def init_crossbreed(self, cand1, cand2):
        cos_coefs = self.combine_lists(cand1.cos_coeffs, cand2.cos_coeffs)
        for coef in cos_coefs:
            self.add_cos_coef(coef)
        sin_coefs = self.combine_lists(cand1.sin_coeffs, cand2.sin_coeffs)
        for coef in sin_coefs:
            self.add_sin_coef(coef)

    def combine_lists(self, l1, l2):
        coefs = []
        for i in range(max(len(l1), len(l2))):
            if i < len(l1) and i < len(l2):
                coef1 = l1[i]
                coef2 = l2[i]
                coefs.append((coef1 + coef2) / 2)
            elif i < len(l1) and i >= len(l2):
                coefs.append(l1[i])
            elif i >= len(l1) and i < len(l2):
                coefs.append(l2[i])
        return coefs

    def mute_first_coef(self):
        self.first_coef += random.uniform(-1, 1)


