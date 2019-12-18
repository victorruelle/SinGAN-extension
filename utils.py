from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig("config.json")


@manager.capture
def test(input_dir,*args,**kwargs):
    print(input_dir)
    print(args)
    print(kwargs)

test()
print(manager.config)
print(test.__dict__)