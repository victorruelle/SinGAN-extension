from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('config.json')

for i in range(50):
    manager.log_scalar("metric",i**2)