from torchvision.models import *
from visualisation.core.utils import device
from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *
import PIL.Image
import cv2
import os

from visualisation.core.utils import device
from visualisation.core.utils import image_net_postprocessing

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing

# for animation

from IPython.display import Image
from matplotlib.animation import FuncAnimation
from collections import OrderedDict


def efficientnet(model_name='efficientnet-b0', **kwargs):
    return EfficientNet.from_pretrained(model_name).to(device)
#     model = EfficientNet.from_pretrained(model_name)
#     state_dict = torch.load("best_checkpoint.pth")["net"]

#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     # load params
#     model.load_state_dict(new_state_dict)
#     model.to(device)
#     return model


max_img = 1
# path = r'D:/data/bendV3-800-2/Train'
# interesting_categories = ['OK', 'NG']
path = r'D:\data\efvistest'
interesting_categories = ['bear']

images = []
image_names = []
for category_name in interesting_categories:
    img_names = os.listdir(os.path.join(path,category_name))
    image_names.extend(img_names)
    image_paths = glob.glob(f'{path}/{category_name}/*')
#     category_images = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
    category_images = list(map(lambda x: PIL.Image.open(x), image_paths))
    images.extend(category_images)

inputs = [Compose([Resize((224, 224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in
          images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

model_outs = OrderedDict()
# model_instances = [alexnet, densenet121,
#                    lambda pretrained: efficientnet(model_name='efficientnet-b0'),
#                    lambda pretrained: efficientnet(model_name='efficientnet-b4')]

model_instances = [lambda pretrained: efficientnet(model_name='efficientnet-b0')]

model_names = [m.__name__ for m in model_instances]
# model_names[-2], model_names[-1] = 'EB0', 'EB4'
model_names[0]= 'EB0'
print(model_names)
print(model_instances)
images = list(map(lambda x: cv2.resize(np.array(x), (224, 224)), images))  # resize i/p img

for name, model in zip(model_names, model_instances):
    # print("s12")
    print(name)
    module = model(pretrained=True).to(device)
    module.eval()

    vis = GradCam(module, device)
    print(vis)
    model_outs[name] = list(map(lambda x: tensor2img(vis(x, None, postprocessing=image_net_postprocessing)[0]), inputs))
    del module
    torch.cuda.empty_cache()


# #### only write into diss
# for index in range(len(images)):
#     cv2.imwrite("tes.jpg",(model_outs[model_names[0]][index] * 255).astype(np.uint8))



####plot and save
img_save_path = "./images/output"
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
axes = [ax2]
for index in range(len(images)):
    a =model_outs["EB0"][index]
    ax1.imshow(images[index])
    ax2.imshow(model_outs["EB0"][index])
    # cv2.imwrite("tes.jpg",(model_outs["EB0"][0] * 255).astype(np.uint8))
    new_im = PIL.Image.fromarray((model_outs["EB0"][index] * 255).astype('uint8'))
    new_im.save("{}/{}_{}.jpg".format(img_save_path, image_names[index], "GCAM"))
    plt.show() # 图3


### save as gif ，single model
# def update(frame):
#     all_ax = []
#     ax1.set_yticklabels([])
#     ax1.set_xticklabels([])
#     ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top", fontsize=30)
#     all_ax.append(ax1.imshow(images[frame]))
#     for i, (ax, name) in enumerate(zip(axes, model_outs.keys())):
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.text(1, 1, name, color="white", ha="left", va="top", fontsize=20)
#         all_ax.append(ax.imshow(model_outs[name][frame], animated=True))
#
#     return all_ax
#
#
# ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
# model_names = [m.__name__ for m in model_instances]
# model_names = ', '.join(model_names)
# fig.tight_layout()
# ani.save('./my_arch.gif', writer='imagemagick')


#### multiple models compare
#### create a figure with two subplots
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 20))
# axes = [ax2, ax3, ax4, ax5]
#
#
# def update(frame):
#     all_ax = []
#     ax1.set_yticklabels([])
#     ax1.set_xticklabels([])
#     ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top", fontsize=30)
#     all_ax.append(ax1.imshow(images[frame]))
#     for i, (ax, name) in enumerate(zip(axes, model_outs.keys())):
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.text(1, 1, name, color="white", ha="left", va="top", fontsize=20)
#         all_ax.append(ax.imshow(model_outs[name][frame], animated=True))
#
#     return all_ax
#
#
# ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
# model_names = [m.__name__ for m in model_instances]
# model_names = ', '.join(model_names)
# fig.tight_layout()
# ani.save('../compare_arch.gif', writer='imagemagick')

