from brarista_ml.brarista_ml.interfaces import TrainingInterface

from diffusion.dataloader import dataloader
from diffusion.unet import SimpleUnet
from diffusion.dataloader import get_data_loader, StanfordCars


# import torchvision
import os
import torch
# import ttools
import yaml
from multiprocessing import cpu_count

from torch.utils.tensorboard import SummaryWriter

# from shutil import copyfile


from brarista_ml.brarista_ml.utils import builder
from brarista_ml.brarista_ml.interfaces.interfaces import TrainingInterface
from brarista_ml.brarista_ml.trainers.custom_trainers import GeneralTrainer
# from brarista_ml.interfaces.evaluator import AccuracyEvaluator

import argparse


num_workers = cpu_count()
# # num_workers = 1

torch.manual_seed(42)


def training_setup(args):
    config = builder.load_yaml(args.config_file)
    
    train_root_path = "data/cars_train"
    test_root_path = "data/cars_test"
    dataloader = get_data_loader(train_root_path, test_root_path)

    training_config = config['TRAINING']

    model_name = training_config['model_name']
    version = training_config['version']
    model_params = training_config.get('MODEL_PARAMS')
    model_params = model_params if model_params is not None else {}
    model = SimpleUnet(**model_params)
    optimizer = builder.build_optimizer(training_config['OPTIMIZER'], [{"params": model.parameters()}])

    interface = TrainingInterface(model, training_config, optimizer, torch.nn.BCEWithLogitsLoss())

    trainer = GeneralTrainer(interface, metric = 'accuracy', load_from_checkpoint=True)

    keys = ["loss"]

    model_checkpoint_dir = os.path.join(os.path.join(training_config["checkpoint_directory"], f"{model_name}_{version}"))
    writer = SummaryWriter(os.path.join(model_checkpoint_dir, "train_summaries"), flush_secs=1)
    val_writer = SummaryWriter(os.path.join(model_checkpoint_dir, "val_summaries"), flush_secs=1)
    train_batches = len(dataloader)
    

#     trainer.add_callback(ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer, val_writer=val_writer, val_keys = keys, frequency=3))
#     trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys, val_keys=keys))

#     copyfile(args.config_file, os.path.join(os.path.join(model_checkpoint_dir, os.path.basename(args.config_file))))

    trainer.train(dataloader, num_epochs = training_config["epochs"], starting_epoch = interface.epoch_num, log_dir=model_checkpoint_dir)

    
    with open(os.path.join(model_checkpoint_dir, os.path.basename(args.config_file)), 'w') as f:
        yaml.dump(config, f)
    

# def evaluate_model(ts_file, checkpoint_file, config_file):
#     config = builder.load_yaml(config_file)
#     training_config = config['TRAINING']
#     model_params = list(training_config['MODEL_PARAMS'].values())
#     if ts_file:
#         model = torch.jit.load(ts_file)
#     else:
#         model = LateFusionClassifier(*model_params)
#         checkpoint = torch.load(checkpoint_file)
#         model.load_state_dict(checkpoint["model_state_dict"])

#     test_dataloader = get_dataloader(config['DATA']["TESTING"])
    
#     evaluator  = AccuracyEvaluator(model)
#     accuracy = evaluator.evaluate(test_dataloader)
    
#     print(f"Report: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default = "training_configs/config.yml", type=str)
    parser.add_argument("--checkpoint_file", required=False, type=str)
    parser.add_argument("--ts_file", required=False, default=None, type=str)
    parser.add_argument("--evaluate", action = "store_true", dest="evaluate")

    args = parser.parse_args()

    training_setup(args)
    # if args.evaluate:
    #     evaluate_model(args.ts_file, args.checkpoint_file, args.config_file)
    # else:
    #     training_setup(args)
