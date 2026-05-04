import numpy as np
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

def setupDataset(FLFMDataset,args,subimage_shape,img_shape,n_threads):
    # dataset = FLFMDataset(args.data_folder, args.data_folder_vol, args.lenslet_file, 
    #                   subimage_shape, img_shape,args.psf_label,args.data_scale, 
    #                   images_to_use=args.images_to_use, n_depths_to_fill=args.n_depths,load_vols=args.load_vols)
    # dataset_test = FLFMDataset(args.data_folder_test, args.data_folder_vol_test, args.lenslet_file, 
    #                         subimage_shape, img_shape,args.psf_label,args.data_scale, 
    #                         images_to_use=args.images_to_use_test, n_depths_to_fill=args.n_depths,load_vols=args.load_vols)
    dataset = FLFMDataset(args.data_folder, args.data_folder_vol, args.lenslet_file, 
                      subimage_shape, img_shape,args.data_scale, 
                      images_to_use=args.images_to_use, n_depths_to_fill=args.n_depths,load_vols=args.load_vols)
    dataset_test = FLFMDataset(args.data_folder_test, args.data_folder_vol_test, args.lenslet_file, 
                            subimage_shape, img_shape,args.data_scale, 
                            images_to_use=args.images_to_use_test, n_depths_to_fill=args.n_depths,load_vols=args.load_vols)
    
    print("# Using ", args.n_depths, " depths, output shape: ", args.output_shape)

    if args.load_vols:
        max_images,max_volumes = dataset.get_max() 
        mean_imgs,std_images,mean_vols,std_vols = dataset.get_statistics()
        stats = {'norm_type':args.norm_type, 'norm_type_img':args.norm_type, 
                'mean_imgs':mean_imgs, 'std_images':std_images, 'max_images':max_images,
                'mean_vols':mean_vols, 'std_vols':std_vols, 'max_vols':max_volumes}
        print("# Mean volumes: ", mean_vols, "\n  std volumes: ", std_vols)
    else:
        max_images = dataset.get_max() 
        mean_imgs,std_images = dataset.get_statistics()
        stats = {'norm_type':args.norm_type, 'norm_type_img':args.norm_type, 
                'mean_imgs':mean_imgs, 'std_images':std_images, 'max_images':max_images} 
    print("# Max images: ", max_images, "Mean images: ", mean_imgs, "\n  std images: ", std_images)

    dataset_size = len(dataset)
    print("# Dataset size: ", dataset_size)

    indices = list(range(dataset_size))
    split = int(np.ceil(args.validation_split * dataset_size))
    if args.shuffle_dataset :
        # np.random.seed(261290)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print("# Training size: ", len(train_indices), ", validation size: ", len(val_indices))
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    # Set up data loaders
    data_loaders = {\
        'train' :data.DataLoader(dataset, batch_size=args.batch_size,sampler=train_sampler, pin_memory=False, num_workers=n_threads), \
        'val'   :data.DataLoader(dataset, batch_size=args.batch_size,sampler=valid_sampler, pin_memory=False, num_workers=n_threads), \
        'test'  :data.DataLoader(dataset_test, batch_size=1, pin_memory=False, num_workers=n_threads, shuffle=True) \
        }

    return dataset, dataset_test, data_loaders, stats