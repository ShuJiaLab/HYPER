# The writer object is used in the training loop to log various metrics and images.
import torch
from torch.utils.tensorboard import SummaryWriter
import zipfile

def setupWriter(args, save_folder, params, net, graphinput,XMLFILENAME):
    writer = SummaryWriter(log_dir=save_folder)
    writer.add_text('arguments',str(vars(args)),0)
    writer.flush()
    writer.add_scalar('params/', params)
    # scripted_net = torch.jit.script(net)
    # writer.add_graph(scripted_net, graphinput)
    # writer.add_graph(net, graphinput)
    # Store files
    zf = zipfile.ZipFile(save_folder + "/files.zip", "w")
    for ff in args.files_to_store:
        zf.write(ff)
    zf.write(XMLFILENAME)
    zf.close()
    print("# Files stored in ", save_folder + "/files.zip")
    return writer