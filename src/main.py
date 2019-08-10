from dataset import *
from wgan import *
from generators import *
from critics import *

dataset = PianoRollData(img_size=64)
# dataset = MNISTData()

generator = DCGANGenerator(img_size=dataset.img_size,
                        channels=dataset.channels)
critic = DCGANCritic(img_size=dataset.img_size,
                  channels=dataset.channels)

wgan = WGAN(generator=generator,
            critic=critic,
            dataset=dataset,
            z_size=100)

wgan(batch_size =16, steps=20000, model_path=project_path.model_path)
