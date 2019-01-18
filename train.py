#!/g/kreshuk/matskevych/environment/miniconda3/envs/neuroseg/bin/python3.6

import logging
import sys
import argparse
import os

from neuro_loader import get_neuro_loaders

from inferno.utils.io_utils import yaml2dict
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.io.transform.base import Compose
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.essentials import DumpHDF5Every

from neurofire import models
from inferno.trainers.basic import Trainer
from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import ApplyAndRemoveMask, RemoveSegmentationFromTarget, InvertTarget


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up and return the `Trainer`, the main inferno class that handles training
def set_up_training(project_directory, config, use_affinities):

    # Load the model to train from the configuratuib file ('./config/train_config.yml')
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    # Initialize the loss: we use the SorensenDiceLoss, which has the nice property
    # of being fairly robust for un-balanced targets
    criterion = SorensenDiceLoss()
    # Wrap the loss to apply additional transformations before the actual
    # loss is applied. Here, we apply the mask to the target
    # and invert the target (necessary for sorensen dice) during training.
    # In addition, we need to remove the segmentation from the target
    # during validation (we only keep the segmentation in the target during validation)

    if use_affinities:
        loss_train = LossWrapper(criterion=criterion,
                                 transforms=Compose(ApplyAndRemoveMask(), InvertTarget()))
    else:
        loss_train = LossWrapper(criterion=criterion,
                                 transforms=Compose(InvertTarget()))

    loss_val = LossWrapper(criterion=criterion,
                           transforms=Compose(InvertTarget()))

    logger.info("Building trainer.")
    smoothness = 0.95
    # Build the trainer object
    trainer = Trainer(model)\
        .save_every((1000, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss_train)\
        .build_validation_criterion(loss_val)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((200, 'iterations'), for_num_iterations=50)\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))\
        .build_metric(criterion)\
        .register_callback(AutoLR(factor=0.98,
                                  patience='200 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))\
        .register_callback(DumpHDF5Every(frequency='200 iterations',
                                         to_directory=os.path.join(project_directory, 'debug')))

    logger.info("Building logger.")
    # Build tensorboard logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(200, 'iterations')).observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard,
                         log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


# Run the training
def training(project_folder,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)
    trainer = set_up_training(project_folder, config, yaml2dict(data_configuration_file).get("master_config").
                              get("use_affinities", True))

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_neuro_loaders(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_neuro_loaders(validation_configuration_file)

    trainer.set_max_num_iterations(max_training_iters)

    # Bind training and validation loader to the trainer
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def main():
    # Input arguments: the project directory to save weights etc
    # and the number of iterations to train
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, default="/g/kreshuk/matskevych/boundary_map_prediction/project_folder/")
    parser.add_argument('--config_name', type=str, default="affinities_with_trafo")
    parser.add_argument('--config_folder', type=str, default="./configs/")  # current_path = path where sBATCH file is
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))

    args = parser.parse_args()

    project_folder = args.project_folder
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    config_dir = os.path.join(args.config_folder, args.config_name)
    train_config = os.path.join(config_dir, 'train_config.yml')
    data_config = os.path.join(config_dir, 'data_config.yml')
    validation_config = os.path.join(config_dir, 'validation_config.yml')

    training(project_folder,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters)


if __name__ == '__main__':
    main()
