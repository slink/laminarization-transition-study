# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


from abc import ABC, abstractmethod

class PDEModel(ABC):
    def __init__(self, params=None):
        self.params = params

    @abstractmethod
    def rhs(self, u, t):
        pass
