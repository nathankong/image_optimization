import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from model import alexnet

torch.manual_seed(0)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:2")
else:
    DEVICE = torch.device("cpu")
print "Device:", DEVICE

# Optimization parameters
MAX_ITER = 400
NEURON_X = 4
NEURON_Y = 3
CHAN = 100
BATCH_SIZE = 1
WEIGHT_DECAY = 0.001
EPS = 1e-2
LR = 0.1
LAYER_IDX = 6
IMG_SIZE = 128

def loss_func(activation):
    return -1. * activation

# DISCLAIMER: CURRENTLY ONLY OPTIMIZING FOR NEURONS IN CONV_1 OF ALEXNET
def main():
    a = alexnet(pretrained=True, to_select=[LAYER_IDX]).to(DEVICE).eval()
    opt_img = Variable((torch.rand(BATCH_SIZE,3,128,128)).to(DEVICE), requires_grad=True)

    optimizer = optim.SGD([opt_img], lr=LR, weight_decay=WEIGHT_DECAY)
    prev_loss = np.Inf
    opt_images = list()
    opt_images.append(opt_img[0].data.cpu().numpy().transpose(1,2,0))

    losses = list()
    for i in range(MAX_ITER):
        print "Iteration {}: Loss {}".format(i, prev_loss)

        optimizer.zero_grad()

        # Forward prop
        activation = a(opt_img)[LAYER_IDX][0, CHAN, NEURON_X, NEURON_Y]

        # Compute loss
        loss = loss_func(activation)
        curr_loss = loss.data.cpu().numpy()

        # Back propagation
        loss.backward()
        optimizer.step()

        # Save image. Index is 0 since only optimizing for one image right now
        t = opt_img[0].data.cpu().numpy().transpose(1,2,0)
        opt_images.append(t)

        # Check convergence
        if np.abs(curr_loss - prev_loss) <= EPS:
            break
        losses.append(curr_loss)
        prev_loss = curr_loss

    return opt_images, losses

if __name__ == "__main__":
    imgs, losses = main()
    print imgs[0].shape
    np.save("images/iter_images.npy", np.array(imgs))
    np.save("losses/losses.npy", np.array(losses))


