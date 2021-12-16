# vision-dqn
## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
poetry install
```
To run our Resnet model on \<game\> [Breakout, SpaceInvaders, MsPacman]:
```bash
poetry run python dqn_atari.py --exp-name=<game>_resnet --encoder resnet --gym-id <game>NoFrameskip-v4
```

To run DCGAN model on  \<game\> [Breakout, SpaceInvaders, MsPacman]:
```bash
poetry run python <dcgan_pre_training>
poetry run python dqn_atari.py --exp-name=<game>_resnet --encoder dcgan --dcgan_path <dcgan_path> --gym-id <game>NoFrameskip-v4
```
