from argparse import Namespace
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

def get_args_parser(add_help: bool = True) -> Namespace:
    import argparse
    parser = argparse.ArgumentParser(description="CDDOD Traning", add_help=add_help)
    parser.add_argument("--config", default=None, type=str, help="Path to the config file")
    parser.add_argument('--wandb', action='store_true', help="whether to use weight and biases")
    parser.add_argument("--project_name", default=None, type=str, help="project name for weight and bias")
    parser.add_argument("--experiment_name", default=None, type=str, help="experiment name in a project for weight and bias")
    parser.add_argument("--group", default=None, type=str, help="group name for weight and bias experiments")
    parser.add_argument("--job_type", default=None, type=str, help="job type e.g. train/test/evaluation")
    parser.add_argument('--fpa', action='store_true', help="Use feature pyramid alignment in the model architecture")
    parser.add_argument('--ra', action='store_true', help="Use region alignment in the model architecture")
    parser.add_argument('--strong_augment', action='store_true', help="Use augmentations on source dataset")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--distributed", action="store_true", help="for data parallelism training")
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="seed for the experiment, reproducibility")
    parser.add_argument("--root_dir_source", default="/data/doc_chs_median_release", type=str, help="source data path")
    parser.add_argument("--root_dir_target", default="/data/wpi", type=str, help="target dataset path")
    parser.add_argument("--source_img_extension", default="png", type=str, help="image extension of the source")
    parser.add_argument("--target_img_extension", default="jpg", type=str, help="image extension of the target")
    parser.add_argument("--source_txt_file", default="trainval", type=str, help="txt file with source images' names")
    parser.add_argument("--target_txt_file", default="train", type=str, help="text file with target images' names")
    parser.add_argument("--target_eval_txt_file", default="test", type=str, help="txt file target images'names-testing")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--job_type", default="train", type=str, help="training or testing")
    parser.add_argument("--lambda1", default=0.1, type=float, help="controls trade-off between detection and fpa loss")
    parser.add_argument("--lambda2", default=0.1, type=float, help="controls trade-off between detection and ra loss")
    parser.add_argument("--epochs", default=15, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--print_freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output_dir", default="/logs", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=0, type=int)
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--batch_size", default=2, type=int,
        help="images per gpu, total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
        help="weight decay (default: 1e-4)", dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay", default=None, type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=5, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps", default=[5, 10], nargs="+", type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--wandb_token", default=None, type=str, help="Token for wandb to upload logs, works only if it's set"
    )
    parser.add_argument(
        "--classes", default=('__background__', 'text', 'listitem', 'heading', 'table', 'figure'),
        type=tuple, help="class names of the labels")
    args = parser.parse_args()
    return args


def read_config_file(args: Namespace):
    """
    overrides the default arguments from config yaml file.
    Args:
        args (Namespace): parsed arguments
    """
    config_file_path = args.config
    if config_file_path is not None and Path(config_file_path).is_file():
        logger.info(f'CONFIG FILE {config_file_path}, WILL BE USED')
        try:
            with open(config_file_path, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                for key in config.keys():
                    for k, v in config[key].items():
                        setattr(args, k.lower(), v)
        except yaml.YAMLError as exc:
            logger.info(exc)
    else:
        logger.info('NO ARGUMENT WAS PASSED FOR A CONFIG FILE, WILL RUN WITH (DEFAULT) ARGS')
