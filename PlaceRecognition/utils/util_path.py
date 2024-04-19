import os

from setup import config

CHECKPOINT = os.path.join(config.base_dir, config.checkpoint_subdir)
VISUALIZATION = os.path.join(config.base_dir, config.vis_subdir)
EVALUATION = os.path.join(config.base_dir, config.eval_subdir)