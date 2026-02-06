# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import json
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def save_run(result, path):
    with open(path, "w") as f:
        json.dump(result, f, cls=_NumpyEncoder, indent=2)


def load_run(path):
    with open(path, "r") as f:
        return json.load(f)
