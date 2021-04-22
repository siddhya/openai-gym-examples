import collections.abc
import json
import networkx
import matplotlib.pyplot as plt
import time

class obj():
    def __init__(self):
        self.options = {
            "env": {
                "type": {
                    ["non-associative", "associative", "mdp"]
                },
                "dynamics": {
                    "known": None,
                    "unknown": None,
                    "changing": None
                },
                "obs-space": {
                    "discrete": None,
                    "continous": None
                },
                "action-space": {
                    "discrete": None,
                    "continous": None
                }
            },
            "model": {
                "type": {
                    "tabular": None,
                    "fn-approx": {
                        "linear": None,
                        "non-linear": {
                            "polynomial": None,
                            "tile-coding": None,
                            "dnn": None
                        }
                    }
                },
                "update": {
                    "absolute": None,
                    "bootstrap": None
                },
                "learning": {
                    "on-policy": None,
                    "off-policy": None
                },
                "value": {
                    "state": None,
                    "action": None,
                    "policy-grad": None
                },
            },
            "runloop": {
                "learning-rate": None,
                "update-interval": None,
                "batch-size": None
            }
        }
        self.data = {}

    def set(self, d):
        if self.compare_dicts(d, self.options):
            return self.update_dict(self.data, d)
        print(f"Wrong values provided: {d}")

    def unset(self, d):
        if self.compare_dicts(d, self.data):
            return self.remove_dict(self.data, d)
        print(f"Wrong values provided: {d}")

    def print(self):
        if self.data:
            print(json.dumps(self.data, indent = 4, default = str))

    # https://stackoverflow.com/a/57657881
    def compare_dicts(self, a, b):
        for key, value in a.items():
            if key in b:
                if isinstance(a[key], dict):
                    if not self.compare_dicts(a[key], b[key]):
                        return False
                elif value != b[key]:
                    return False
            else:
                return False
        return True

    # https://stackoverflow.com/a/3233356
    def update_dict(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self.update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def remove_dict(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self.update_dict(d.get(k, {}), v)
            else:
                del d[k]
                #d[k] = v
        return d

    def load(self, f):
        with open(f, "r") as fp:
            self.data = json.load(fp)
    
    def dump(self, f):
        with open(f, "w") as fp:
            self.data = json.dump(self.data, fp)

#o = obj()
#o.load("mc.json")
#o.unset({"env": {"type": {"mdp": None}}})
#o.set({"env": {"type": {"associative": None}}})
# o.set({"env": {"dynamics": {"unknown": None}}})
# o.set({"env": {"obs-space": {"continous": None}}})
# o.set({"env": {"action-space": {"continous": None}}})
# o.set({"model": {"type": {"tabular": None}}})
# o.set({"model": {"update": {"absolute": None}}})
# o.set({"model": {"learning": {"off-policy": None}}})
# o.set({"model": {"value": {"action": None}}})
# o.dump("mc.json")
#o.print()

d = {
            "env": {
                "type": {
                    "non-associative": None,
                    "associative": None,
                    "mdp": None
                },
                "dynamics": {
                    "known": None,
                    "unknown": None,
                    "changing": None
                },
                "obs-space": {
                    "discrete": None,
                    "continous": None
                },
                "action-space": {
                    "discrete": None,
                    "continous": None
                }
            },
            "model": {
                "type": {
                    "tabular": None,
                    "fn-approx": {
                        "linear": None,
                        "non-linear": {
                            "polynomial": None,
                            "tile-coding": None,
                            "dnn": None
                        }
                    }
                },
                "update": {
                    "absolute": None,
                    "bootstrap": None
                },
                "learning": {
                    "on-policy": None,
                    "off-policy": None
                },
                "value": {
                    "state": None,
                    "action": None,
                    "policy-grad": None
                },
            },
            "runloop": {
                "learning-rate": None,
                "update-interval": None,
                "batch-size": None
            }
        }
g = networkx.convert.from_dict_of_dicts(d)
#graph = networkx.DiGraph()
# graph.add_edge("RL", "env")
# graph.add_edge("RL", "model")
# graph.add_edge("RL", "runloop")

# graph.add_edge("env", "type")
# graph.add_edge("env", "dynamics")
# graph.add_edge("env", "obs-space")
# graph.add_edge("env", "action-space")

# graph.add_edge("model", "type")
networkx.drawing.nx_pylab.draw(graph)
plt.show()
print("End")