import pytorch_lightning as pl
import argparse
import time
import os

from model import PedalNet

def gen_timestamp_name() -> str:
    """generate a timestap to use as filename"""
    secondsSinceEpoch = time.time() # Get the seconds since epoch 
    timeObj = time.localtime(secondsSinceEpoch) # Convert seconds since epoch to struct_time
    name = '%04d.%02d.%02d-%02d%02d%02d' % (
    timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
    return name

def main(args):

    model = PedalNet(vars(args))

    version = gen_timestamp_name()
    version += "_"+args.name
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="",version=version)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        gpus=None if args.cpu else args.gpus,
        logger=logger, 
        log_every_n_steps=100
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--dilation_depth", type=int, default=8)
    parser.add_argument("--num_repeat", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1_500)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--name", default="default")

    args = parser.parse_args()
    main(args)
