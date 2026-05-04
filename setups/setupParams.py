                                          
from xml.etree.ElementTree import Element, SubElement, ElementTree
import argparse
import ast

parser = argparse.ArgumentParser()
args = parser.parse_args()

def defDefault():
    parser.add_argument('--data_folder', nargs='?', default= '../../dldatasets/train/syn_in')
    parser.add_argument('--data_folder_vol', nargs='?', default= '../../dldatasets/train/syn_gt')
    parser.add_argument('--data_folder_test', nargs='?', default='../../dldatasets/test/syn_in')
    parser.add_argument('--data_folder_vol_test', nargs='?', default='../../dldatasets/test/syn_gt')
    parser.add_argument('--lenslet_file', nargs='?', default= "lenslet_centers_python_3.txt")
    parser.add_argument('--files_to_store', nargs='+', default=['mainTrainFLFMNet.py',
                                                                'dataprep/FLFMDataset.py',
                                                                'utils/misc_utils.py',
                                                                'nets/extra_nets.py',
                                                                'nets/FLFMnet.py'])
    parser.add_argument('--psf_file', nargs='?', default= "PSFFLFint_Sim65nm_20200320_Blue_gly_10um1025.mat")
    parser.add_argument('--prefix', nargs='?', default= "test")
    parser.add_argument('--checkpoint', nargs='?', default= "")
    parser.add_argument('--checkpoint_FLFMnet', nargs='?', default= "")

    # Set up data range
    parser.add_argument('--images_to_use_start', nargs='+', type=int, default=1)
    parser.add_argument('--images_to_use_end', nargs='+', type=int, default=50)
    parser.add_argument('--images_to_use_test_start', nargs='+', type=int, default=1)
    parser.add_argument('--images_to_use_test_end', nargs='+', type=int, default=10)
    parser.add_argument('--n_depths', type=int, default= 16) # The n_depths cannot be scaled by data_scale.

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--shuffle_dataset', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--use_bias', type=int, default=0)
    parser.add_argument('--data_scale', type=float, default=[1,1], help='resize dataset spatial dim by this factor')
    parser.add_argument('--loss_type', type=str, default='l2', 
        help='Define loss types: l1, l2, smoothl1, ssim, msssim,ssiml1,msssiml1,ssiml2,msssiml2')
    parser.add_argument('--loss_z_smooth_lambda', type=float, default=0.01)
    
    # Noise arguments
    parser.add_argument('--add_noise', type=int, default=0, help='Apply noise to images? 0 or 1')
    parser.add_argument('--signal_power_min', type=float, default=1, 
                        help='Min signal value to control signal to noise ratio when applyting noise.')
    parser.add_argument('--signal_power_max', type=float, default=10, 
                        help='Max signal value to control signal to noise ratio when applyting noise.')
    parser.add_argument('--norm_type', type=float, default=1, 
                        help='Normalization type, see the normalize_type function for more info.')
    
    parser.add_argument('--corrected_Xc_file', nargs='?', default='',
                        help='Path to .mat file containing corrected_Xc displacement data')
    parser.add_argument('--dark_current', type=float, default=106, help='Dark current value of camera.')
    parser.add_argument('--dark_current_vol', type=float, default=0, help='Dark current value of GT.')

    parser.add_argument('--use_sparse', type=int, default=0)
    parser.add_argument('--use_img_loss', type=float, default=1.0)

    parser.add_argument('--unet_depth', type=int, default=2)
    parser.add_argument('--unet_wf', type=int, default=7)
    parser.add_argument('--unet_drop_out', type=float, default=0.0)

    parser.add_argument('--output_path', nargs='?', default='./checkpoints/')
    parser.add_argument('--main_gpu', nargs='+', type=int, default=[0])
    parser.add_argument('--gpu_repro', nargs='+', type=int, default=[0])
    parser.add_argument('--n_split', type=int, default=1)

    args = parser.parse_args()
    return args

def pretty_xml(element, indent, newline, level=0):
    
    if element:
        if (element.text is None) or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  
            subelement.tail = newline + indent * level
        if subelement.text == '':
            subelement.text = '""'
        pretty_xml(subelement, indent, newline, level=level + 1)

def write2XML(xmlfilename, args):
    args = parser.parse_args()
    args_dict = vars(args)
    root = Element('train_params')
    for key, value in args_dict.items():
        child = SubElement(root, key)
        child.attrib['type'] = str(type(value).__name__)
        child.text = str(value)
    pretty_xml(root, '\t', '\n')
    tree = ElementTree(root)
    tree.write(xmlfilename, encoding='utf-8', xml_declaration=True)

def getArgsFromXML(xmlfilename):
    args = parser.parse_args()
    tree = ElementTree(file=xmlfilename)
    root = tree.getroot()
    for child in root:
        childvalue = child.text
        if child.attrib['type'] == 'int':
            childvalue = int(childvalue)
        elif child.attrib['type'] == 'float':
            childvalue = float(childvalue)
        elif child.attrib['type'] == 'list':
            childvalue = ast.literal_eval(childvalue)
        elif child.attrib['type'] == 'bool':
            # Proper boolean parsing: only "True" and "1" are True, everything else is False
            childvalue = childvalue.strip().lower() in ('true', '1')
        elif child.attrib['type'] == 'str':
            if childvalue == '""':
                childvalue = ''
            else:
                childvalue = childvalue
        setattr(args, child.tag, childvalue)
    if hasattr(args, 'images_to_use_start') and hasattr(args, 'images_to_use_end'):
        args.images_to_use = range(args.images_to_use_start-1, args.images_to_use_end)
    if hasattr(args, 'images_to_use_test_start') and hasattr(args, 'images_to_use_test_end'):
        args.images_to_use_test = range(args.images_to_use_test_start-1, args.images_to_use_test_end)
    return args

def setupParams(xmlfilename,default=False):
    if default:
        args = defDefault()
        if hasattr(args, 'images_to_use_start') and hasattr(args, 'images_to_use_end'):
            args.images_to_use = range(args.images_to_use_start-1, args.images_to_use_end)
        if hasattr(args, 'images_to_use_test_start') and hasattr(args, 'images_to_use_test_end'):
            args.images_to_use_test = range(args.images_to_use_test_start-1, args.images_to_use_test_end)
        write2XML(xmlfilename, args)
    else:
        args = getArgsFromXML(xmlfilename)
    return args

if __name__ == '__main__':
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    args = setupParams('./trainingparameters/default.xml',default=False)
    print(args)
    