from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig("config.json")


def show_ops(ops):
    for k,v in ops.__dict__.items():
        if not "__" in k:
            print("{}:{}".format(k,str(v)))