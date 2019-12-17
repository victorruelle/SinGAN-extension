# Keep this on top
from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('config.json')

import tasks

manager.run_queue()