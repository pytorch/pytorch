




import inspect
import logging
logging.basicConfig()
log = logging.getLogger("ModuleRegister")
log.setLevel(logging.DEBUG)

MODULE_MAPS = []


def registerModuleMap(module_map):
    MODULE_MAPS.append(module_map)
    log.info("ModuleRegister get modules from  ModuleMap content: {}".
             format(inspect.getsource(module_map)))


def constructTrainerClass(myTrainerClass, opts):

    log.info("ModuleRegister, myTrainerClass name is {}".
             format(myTrainerClass.__name__))
    log.info("ModuleRegister, myTrainerClass type is {}".
             format(type(myTrainerClass)))
    log.info("ModuleRegister, myTrainerClass dir is {}".
             format(dir(myTrainerClass)))

    myInitializeModelModule = getModule(opts['model']['model_name_py'])
    log.info("ModuleRegister, myInitializeModelModule dir is {}".
             format(dir(myInitializeModelModule)))

    myTrainerClass.init_model = myInitializeModelModule.init_model
    myTrainerClass.run_training_net = myInitializeModelModule.run_training_net
    myTrainerClass.fun_per_iter_b4RunNet = \
        myInitializeModelModule.fun_per_iter_b4RunNet
    myTrainerClass.fun_per_epoch_b4RunNet = \
        myInitializeModelModule.fun_per_epoch_b4RunNet

    myInputModule = getModule(opts['input']['input_name_py'])
    log.info("ModuleRegister, myInputModule {} dir is {}".
             format(opts['input']['input_name_py'], myInputModule.__name__))

    # Override input methods of the myTrainerClass class
    myTrainerClass.get_input_dataset = myInputModule.get_input_dataset
    myTrainerClass.get_model_input_fun = myInputModule.get_model_input_fun
    myTrainerClass.gen_input_builder_fun = myInputModule.gen_input_builder_fun

    # myForwardPassModule = GetForwardPassModule(opts)
    myForwardPassModule = getModule(opts['model']['forward_pass_py'])
    myTrainerClass.gen_forward_pass_builder_fun = \
        myForwardPassModule.gen_forward_pass_builder_fun

    myParamUpdateModule = getModule(opts['model']['parameter_update_py'])
    myTrainerClass.gen_param_update_builder_fun =\
        myParamUpdateModule.gen_param_update_builder_fun \
        if myParamUpdateModule is not None else None

    myOptimizerModule = getModule(opts['model']['optimizer_py'])
    myTrainerClass.gen_optimizer_fun = \
        myOptimizerModule.gen_optimizer_fun \
        if myOptimizerModule is not None else None

    myRendezvousModule = getModule(opts['model']['rendezvous_py'])
    myTrainerClass.gen_rendezvous_ctx = \
        myRendezvousModule.gen_rendezvous_ctx \
        if myRendezvousModule is not None else None

    # override output module
    myOutputModule = getModule(opts['output']['gen_output_py'])

    log.info("ModuleRegister, myOutputModule is {}".
             format(myOutputModule.__name__))
    myTrainerClass.fun_conclude_operator = myOutputModule.fun_conclude_operator
    myTrainerClass.assembleAllOutputs = myOutputModule.assembleAllOutputs

    return myTrainerClass


def overrideAdditionalMethods(myTrainerClass, opts):
    log.info("B4 additional override myTrainerClass source {}".
        format(inspect.getsource(myTrainerClass)))
    # override any additional modules
    myAdditionalOverride = getModule(opts['model']['additional_override_py'])
    if myAdditionalOverride is not None:
        for funcName, funcValue in inspect.getmembers(myAdditionalOverride,
                                                      inspect.isfunction):
            setattr(myTrainerClass, funcName, funcValue)
    log.info("Aft additional override myTrainerClass's source {}".
        format(inspect.getsource(myTrainerClass)))
    return myTrainerClass


def getModule(moduleName):
    log.info("get module {} from MODULE_MAPS content {}".format(moduleName, str(MODULE_MAPS)))
    myModule = None
    for ModuleMap in MODULE_MAPS:
        log.info("iterate through MODULE_MAPS content {}".
                 format(str(ModuleMap)))
        for name, obj in inspect.getmembers(ModuleMap):
            log.info("iterate through MODULE_MAPS a name {}".format(str(name)))
            if name == moduleName:
                log.info("AnyExp get module {} with source:{}".
                         format(moduleName, inspect.getsource(obj)))
                myModule = obj
                return myModule
    return None


def getClassFromModule(moduleName, className):
    myClass = None
    for ModuleMap in MODULE_MAPS:
        for name, obj in inspect.getmembers(ModuleMap):
            if name == moduleName:
                log.info("ModuleRegistry from module {} get class {} of source:{}".
                         format(moduleName, className, inspect.getsource(obj)))
                myClass = getattr(obj, className)
                return myClass
    return None
