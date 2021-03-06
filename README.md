# vision-dqn
## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
poetry install
```
To run our ResNet-based model on \<game\> [Breakout, SpaceInvaders, MsPacman]:
```bash
poetry run python dqn_atari.py --exp-name=<game>_resnet --encoder resnet --gym-id <game>NoFrameskip-v4
```

To run DCGAN-based model on \<game\> [Breakout, SpaceInvaders, MsPacman] (after pretraining DCGAN model):
```bash
poetry run python dqn_atari.py --exp-name=<game>_dcgan --encoder dcgan --dcgan_path <dcgan_path> --gym-id <game>NoFrameskip-v4
```

### Pretraining DCGAN
Open the dcgan-atari-train.ipynb on Google Colab with GPU runtime and run all blocks. It should save checkpoints in the temporary folder on Colab, from which the Discriminator checkpoint path must be assigned as \<dcgan_path\>.
