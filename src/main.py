from dataset import *
from wgan import *
from generators import *
from critics import *

dataset = PianoRollData(img_size=(16,128))
# dataset = MNISTData()

generator = DCGANGenerator(img_size=dataset.img_size,
                        channels=dataset.channels, prev_x = dataset.prev_x)
critic = DCGANCritic(img_size=dataset.img_size,
                  channels=dataset.channels, image = dataset.x)

wgan = WGAN(generator=generator,
            critic=critic,
            dataset=dataset,
            epoches = 1000,
            z_size=100)

wgan(batch_size =64, model_path=project_path.model_path)
