import os
import copy
import time

import configargparse
import yaml

from se3dif.datasets.acronym_dataset import load_train_test_split_files, PartialPointcloudAcronymAndSDFDataset
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader

from se3dif.utils import load_experiment_specifications

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

import matplotlib
matplotlib.use('Agg')


base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))



def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='multiobject_partialp_graspdif'
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=True
                   , help='activate or deactivate summary')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')

    p.add_argument('--device',  type=str, default='cuda',)
    p.add_argument('--class_type', type=str, default='Mug')

    p.add_argument('--allowed_categories', type=str, default=None, help='for using dataset_acronym_shapenetsem')
    p.add_argument('--batch_size', type=int, default=2, help='Batch size')

    opt = p.parse_args()
    return opt


def main(opt):

    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)

    args['TrainSpecs']['batch_size'] = opt.batch_size  # overwrite batch size

    class_type = opt.class_type if opt.allowed_categories is None else opt.allowed_categories

    # saving directories
    root_dir = opt.saving_root
    exp_dir  = os.path.join(root_dir, class_type, args['exp_log_dir'], f"{int(time.time_ns())}")
    args['saving_folder'] = exp_dir

    # create directories
    os.makedirs(exp_dir, exist_ok=False)

    # save options and args
    with open(os.path.join(args['saving_folder'], 'opt.yaml'), 'w') as fp:
        yaml.dump(vars(opt), fp)
    with open(os.path.join(args['saving_folder'], 'args.yaml'), 'w') as fp:
        yaml.dump(args, fp)

    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if opt.allowed_categories is None:
        ## Dataset
        train_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(augmented_rotation=True, one_object=args['single_object'])
        train_dataloader = DataLoader(train_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)
        test_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(augmented_rotation=True, one_object=args['single_object'],
                                                                      test_files=train_dataset.test_grasp_files)
        test_dataloader = DataLoader(test_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)
    else:
        train_files, test_files = load_train_test_split_files(
            opt.allowed_categories, os.path.join(get_root_src(), '..', 'dataset_acronym_shapenetsem')
        )

        # Instantiate the dataset with custom train files
        train_dataset = PartialPointcloudAcronymAndSDFDataset(
            n_pointcloud=1024,
            visualize=False,
            augmented_rotation=True,
            one_object=False,
            phase='train',
            use_split_files=True,
            train_files=train_files
        )

        # Instantiate the dataset with custom test files
        test_dataset = PartialPointcloudAcronymAndSDFDataset(
            n_pointcloud=1024,
            visualize=False,
            augmented_rotation=True,
            one_object=False,
            phase='test',
            use_split_files=True,
            test_files=test_files
        )

        # Create DataLoaders
        data_loader_options = {}
        data_loader_options['num_workers'] = 0
        data_loader_options['pin_memory'] = True
        data_loader_options['persistent_workers'] = data_loader_options['num_workers'] > 0

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True,
            **data_loader_options)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=False,
            **data_loader_options)


    ## Model
    args['device'] = device
    model = loader.load_model(args)

    # Losses
    loss = losses.get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn

    ## Summaries
    summary = summaries.get_summary(args, opt.summary)

    ## Optimizer
    lr_schedules = get_learning_rate_schedules(args)
    optimizer = torch.optim.Adam([
            {
                "params": model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": model.decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])

    # Train
    trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir= exp_dir,
                summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                val_dataloader=test_dataloader
                  )


if __name__ == '__main__':
    args = parse_args()
    main(args)