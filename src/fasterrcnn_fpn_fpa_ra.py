import os

from get_parser import read_config_file, get_args_parser

# ======================== FOR DEBUGGING ========================================
if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    # please change to your port number
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        'localhost',
        port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', "12053")),
        stdoutToServer=True,
        stderrToServer=True,
        suspend=False
    )
# ===============================================================================
import gc
import time
import torch
import datetime
# cddod utils
import dod_utils as cddod
from dod_modules import Resnet101withFPN
from train_one_epoch import train_one_epoch
from dod_custom_faster_rcnn import FasterRCNNcddod
from pytorch_references.detection.engine import evaluate


def main():
    args = get_args_parser()

    if args.output_dir:
        cddod.mkdir(args.output_dir)

    if args.test_only:
        logger = cddod.make_logger(filename='testing.log', file_dir=args.output_dir)
    else:
        logger = cddod.make_logger(filename='training.log', file_dir=args.output_dir)

    read_config_file(args)

    wandb_logger = None
    if args.wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_token
        wandb_logger = cddod.wandb_logger(
            project_name=args.project_name,
            name=args.experiment_name,
            job_type=args.job_type,
            group=args.group,
            config=args
        )

    logger.info(args)

    # Setting the experiment in deterministic mode (for reproducibility)
    cddod.set_deterministic(deterministic=True, seed=args.seed)
    # 'CUDA' or 'CPU'
    device = torch.device(args.device)

    logger.info("Loading data and creating datasets")
    # ds: dataset
    source_ds, target_ds, eval_ds = cddod.get_datasets(
        root_dir_source=args.root_dir_source,
        root_dir_target=args.root_dir_target,
        img_extension_s=args.source_img_extension,  # source
        img_extension_t=args.target_img_extension,  # target
        txt_file_s=args.source_txt_file,  # source
        txt_file_t=args.target_txt_file,  # target
        txt_file_eval=args.target_eval_txt_file,
        augment=args.strong_augment,
        classes=args.classes
    )

    logger.info(f'Classes: {args.classes}')
    cddod.print_dataset_len(source_ds, target_ds, eval_ds)
    logger.info("Creating data loaders")

    # dl: dataloader
    source_dl, target_dl, eval_dl = cddod.get_dataloaders(
        dataset_source=source_ds,
        dataset_target=target_ds,
        dataset_eval=eval_ds,
        batch_size=args.batch_size,
        dd_parallel=False
    )

    logger.info("Creating models")

    backbone = Resnet101withFPN().to(device)
    model = FasterRCNNcddod(backbone=backbone, num_classes=len(args.classes), use_fpa=args.fpa, use_ra=args.ra)
    model.to(device)

    # weight decay of model parameters
    parameters = cddod.set_normalized_weight_decay(
        model=model,
        norm_weight_decay=args.norm_weight_decay,
        weight_decay=args.weight_decay
    )

    # optimizer SGD/Adam
    optimizer = cddod.set_optimizer(
        optimizer_name=args.opt,
        parameters=parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = cddod.set_lr_scheduler(
        lr_scheduler=args.lr_scheduler,
        optimizer=optimizer,
        lr_steps=args.lr_steps,
        lr_gamma=args.lr_gamma,
        epochs=args.epochs
    )

    # if one want to resume from a saved checkpoint
    cddod.resume_from_checkpoint(
        resume=args.resume,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        amp=args.amp,
        scaler=scaler,
        args=args
    )

    # for evaluating the model.
    if args.test_only:
        logger.info(f"=============== Testing started ===============")
        torch.backends.cudnn.deterministic = True
        test_stats = evaluate(model, eval_dl, device=device)
        if wandb_logger is not None:
            wandb_logger.log({**test_stats['bbox']})
        # Mean average precision (MAP@0.5)
        logger.info(test_stats['bbox']['(AP) @ IoU=0.50 | area=all | maxDets=100'])
        logger.info(f"=============== Testing finished ===============")

    else:
        logger.info(f"=============== Training started ===============")
        start_time = time.time()

        global_iter_count = 0
        best_map = 0

        for epoch in range(args.start_epoch, args.epochs):

            loss_dict, local_iter_count, _ = train_one_epoch(
                model=model,
                optimizer=optimizer,
                dl_source=source_dl,
                dl_target=target_dl,
                device=device,
                epoch=epoch,
                scaler=scaler,
                wandb_logger=wandb_logger,
                args=args
            )

            lr_scheduler.step()
            global_iter_count += local_iter_count
            # evaluating the model after every epoch
            eval_stats = evaluate(model, eval_dl, device=device)
            if wandb_logger is not None:
                wandb_logger.log({**eval_stats['bbox']})
            # Mean average precision (MAP@0.5)
            current_map = eval_stats['bbox']['(AP) @ IoU=0.50 | area=all | maxDets=100']
            # saving the model
            if current_map > best_map:
                best_map = current_map
                cddod.save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        logger.info(f"Total number of iterations: {global_iter_count}")
        logger.info(f"Training time {total_time_str}")
        logger.info(f"=============== Training finished ===============")


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
