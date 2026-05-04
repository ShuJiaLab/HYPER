from torch import set_num_threads, manual_seed, get_num_threads

def setupDevices(args):
    n_threads = 0
    if len(args.main_gpu)>0:
        device = "cuda:" + str(args.main_gpu[0])
        device_repro = "cuda:" + str(args.main_gpu[0]+1)
    else:
        device = "cpu"
        device_repro = "cuda:0"

    if len(args.gpu_repro)==0:
        device_repro = "cpu"
    else:
        device_repro = "cuda:" + str(args.gpu_repro[0])
    print("# Using device: ", device, " and ", device_repro)

    if n_threads!=0:
        set_num_threads(n_threads)
    manual_seed(261290)
    print("# Using ", get_num_threads(), " threads")
    return device, device_repro,n_threads









