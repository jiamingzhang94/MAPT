import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

# import trainers.coop
# import trainers.cocoop
# import trainers.zsclip
import trainers.maple
# import trainers.independentVL
# import trainers.vpt

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 16  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    # cfg.TRAINER.MAPLE.CTX_INIT = ""  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp32 "  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = args.depth # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.MAPLE.ADV_TRAIN = True
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.MAPLE.EPSILON = 1 / 255
    cfg.TRAINER.MAPLE.TEST_EPSILON = 1. / 255
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.TRAINER.MAPLE.ADV_STEPS = 5
    cfg.TRAINER.MAPLE.TEST_STEPS = 100
    cfg.TRAINER.MAPLE.SURROGATE = "self"
    cfg.TRAINER.MAPLE.FEATURE_CONSTRAIN = args.feature

    # cfg.TRAINER.MAPLE.LAMBDA_CONSIST = 0.1
    # cfg.TRAINER.MAPLE.PROMPT_HIDDEN_DIM = 1024
    # cfg.TRAINER.MAPLE.VISION_HIDDEN_DIM = 2048





def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    # setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    # if args.generate:
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     trainer.generate()

    if args.eval_only:
        # Generate model name based on parameters if not provided
        model_name = args.model_name
        if not model_name and hasattr(cfg.TRAINER, 'MAPLE'):
            # Format epsilon as a string without trailing zeros
            eps_str = str(cfg.TRAINER.MAPLE.EPSILON).rstrip('0').rstrip('.') if cfg.TRAINER.MAPLE.EPSILON % 1 == 0 else str(cfg.TRAINER.MAPLE.EPSILON)
            
            # Include more parameters for better ablation study organization
            depth_str = str(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
            shots_str = str(cfg.DATASET.NUM_SHOTS) if hasattr(cfg.DATASET, 'NUM_SHOTS') else "default"
            
            # Create model name with comprehensive parameter information
            model_name = f"model_eps{eps_str}_steps{cfg.TRAINER.MAPLE.ADV_STEPS}_depth{depth_str}_shots{shots_str}.pth.tar"
        
        trainer.load_model(args.model_dir, epoch=args.load_epoch, model_name=model_name)
        trainer.test()
        trainer.test_adv(args.surrogate)
        return

    if not args.no_train:
        trainer.train()
        trainer.test_adv(args.surrogate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/dycpu6_8tssd1/jmzhang/datasets/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default=False,
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/oxford_flowers.yaml",
        # default="configs/datasets/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    # parser.add_argument("--generate", default=False)
    parser.add_argument("--trainer", type=str, default="MaPLe", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default=False)
    parser.add_argument("--adv-train", default=False)
    parser.add_argument("--surrogate", type=str, default="vanilla")
    parser.add_argument("--depth", type=int, default=12, help="depth")
    parser.add_argument("--eps", type=float, default=1/255.)
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--model-name", type=str, default=None, help="custom model name for loading/saving")
    parser.add_argument("--feature", default=False)



    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
