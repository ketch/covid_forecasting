import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def smooth(x):
    m = 2
    y = x.copy()
    for i in range(1,len(x)-1):
        y[i]=(x[i-1]+m*x[i]+x[i+1])/(m+2)
    return y
