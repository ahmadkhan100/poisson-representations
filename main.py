
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  print("WARNING: no gpu available")

cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.CIFAR10(root="cifar-10", train=True, transform=cifar10_transform_train, download=True)
test_set = datasets.CIFAR10(root="cifar-10", train=False, transform=cifar10_transform_test, download=True)

"""## Sample images"""

num_show = 10
fig, axs = plt.subplots(1, num_show)

for i in range(num_show):
    img, label = train_set[i]
    axs[i].imshow(img.permute(1, 2, 0).detach().numpy())
    axs[i].axis("off")
plt.show()

"""## AlexNet"""

r"""
Adapted from https://github.com/HobbitLong/CMC/blob/f25c37e49196a1fe7dc5f7b559ed43c6fce55f70/models/alexnet.py
"""

import torch.nn as nn


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128):
        super(SmallAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 3 * 3, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x

"""## Train AlexNet"""

BATCH_SIZE = 256
NUM_EPOCHS = 20

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)

alexnet = SmallAlexNet(in_channel=3, feat_dim=128)
net = nn.Sequential(alexnet, nn.Linear(128, 10))
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()

net = net.to(device)

# net.to(device)
total_losses = []
for i in range(NUM_EPOCHS):
    net.train()
    losses = []
    for img, label in tqdm(train_dataloader):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    print("avg loss:", sum(losses) / len(losses))
    total_losses += losses

plt.plot(total_losses)
plt.show()

# torch.save(net.state_dict(), "alexnet-cifar10.pth")
# !cp "alexnet-cifar10.pth" "./drive/MyDrive/cifar-10/"

# !gdown 1-DTOfvJAZJrKJ3bDdgpTdeUu3l9IF18y
# net.load_state_dict(torch.load("alexnet-cifar10.pth"))

net.eval()
accs = []
for img, label in tqdm(train_dataloader):
    img, label = img.to(device), label.to(device)
    pred = net(img).argmax(dim=-1)
    acc = (pred == label).float().mean()
    accs.append(acc.item())
print("train accuracy:", sum(accs) / len(accs))

accs = []
for img, label in tqdm(test_dataloader):
    img, label = img.to(device), label.to(device)
    pred = net(img).argmax(dim=-1)
    acc = (pred == label).float().mean()
    accs.append(acc.item())
print("test accuracy:", sum(accs) / len(accs))

"""## Pass entire dataset through AlexNet"""

def pass_through(net, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # net = net.to(device)
    # net.eval()
    outs = []
    labels = []
    for inps, label in tqdm(dataloader):
        inps = inps.to(device)
        outs.append(net(inps).detach().cpu())
        labels.append(label)
    return {"data": torch.cat(outs, dim=0), "labels": torch.cat(labels, dim=0)}

alexnet.to(device)
alexnet.eval()
alexnet_features_train = pass_through(alexnet, datasets.CIFAR10(
    root="cifar-10",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32) ]))
)
alexnet_features_test = pass_through(alexnet, datasets.CIFAR10(
    root="cifar-10",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32) ]))
)

with open("alexnet_features.bin", "wb") as f:
   pickle.dump({
    "train": {
      k: v.numpy() for k, v in alexnet_features_train.items()
    },
    "test": {
      k: v.numpy() for k, v in alexnet_features_test.items()
    }
  }, f)

!cp "alexnet_features.bin" "./drive/MyDrive/cifar-10/"

"""## or use already passed through values?"""

!gdown 1-BHmANwz6DxNRU7X3MFqJvbRe2HkfJYT

with open("alexnet_features.bin", "rb") as f:
    alexnet_features = pickle.load(f)
alexnet_features = {t: {k: torch.tensor(v) for k, v in dataset.items()} for t, dataset in alexnet_features.items()}

"""## create the dataset"""

class ReprDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features["data"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features["data"][idx], self.features["labels"][idx]

after_alexnet_train = ReprDataset(alexnet_features["train"])
after_alexnet_test = ReprDataset(alexnet_features["test"])

"""## poisson flow utils"""

def forward_pz(samples_batch, m):
    """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      config: configurations
      samples_batch: A mini-batch of un-augmented training data
      m: A 1D torch tensor. The exponents of (1+\tau).

    Returns:
      Perturbed samples
    """
    tau = TAU
    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * SIGMA
    z = z.abs()
    # Confine the norms of perturbed data.
    # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
    # if config.training.restrict_M:
    #     idx = (z < 0.005).squeeze()
    #     num = int(idx.int().sum())
    #     restrict_m = int(M * 0.7)
    #     m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m

    data_dim = HIDDEN_SIZE
    multiplier = (1+tau) ** m

    noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * SIGMA
    norm_m = torch.norm(noise, p=2, dim=1) * multiplier
    # Perturb z
    perturbed_z = z.squeeze() * multiplier
    # Sample uniform angle
    gaussian = torch.randn(len(samples_batch), data_dim).to(device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
    # Construct the perturbation for x
    perturbation_x = unit_gaussian * norm_m[:, None]
    perturbation_x = perturbation_x.view_as(samples_batch)
    # Perturb x
    perturbed_x = samples_batch + perturbation_x
    # Augment the data with extra dimension z
    perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(samples_batch), -1),
                                       perturbed_z[:, None]), dim=1)
    return perturbed_samples_vec

def get_loss_fn():
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `methods.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `Truec` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A PFGM or score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    samples_full = batch
    samples_batch = batch[: SMALL_BATCH_SIZE]

    print(samples_batch.shape)

    m = torch.rand((samples_batch.shape[0],), device=device) * M
    # Perturb the (augmented) mini-batch data
    perturbed_samples_vec = forward_pz(samples_batch, m)     # forward_pz: IMPORT THIS LATER

    # with torch.no_grad():
      # append with a zero
    real_samples_vec = torch.cat(
      (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(device)), dim=1)

    data_dim = HIDDEN_SIZE
    gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                            dim=[-1]).sqrt()

    # For numerical stability, timing each row by its minimum value
    distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
    distance = distance ** (data_dim + 1)
    distance = distance[:, :, None]
    # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
    coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
    diff = - (perturbed_samples_vec.unsqueeze(1) - real_samples_vec)

    # Calculate empirical Poisson field (N+1 dimension in the augmented space)
    gt_direction = torch.sum(coeff * diff, dim=1)
    gt_direction = gt_direction.view(gt_direction.size(0), -1)

    gt_norm = gt_direction.norm(p=2, dim=1)
    # Normalizing the N+1-dimensional Poisson field
    gt_direction /= (gt_norm.view(-1, 1) + GAMMA)
    gt_direction *= np.sqrt(data_dim)

    target = gt_direction

    perturbed_samples_x = perturbed_samples_vec[:, :-1].view_as(samples_batch)
    perturbed_samples_z = torch.clamp(perturbed_samples_vec[:, -1], 1e-10)
    preturbed_samples = torch.cat([perturbed_samples_x, perturbed_samples_z[:,None]], dim=1)
    net = model(preturbed_samples)

    # net_x = net_x.view(net_x.shape[0], -1)
    # Predicted N+1-dimensional Poisson field
    # net = torch.cat([net_x, net_z[:, None]], dim=1)
    loss = ((net - target) ** 2)
    loss = reduce_op(loss.reshape(loss.shape[0], -1), dim=-1)
    loss = torch.mean(loss)

    return loss
  return loss_fn

class PoissonField(nn.Module):
    def __init__(self, dim, num_hidden=2):
        super().__init__()
        self.dim = dim
        self.num_hidden = num_hidden

        self.fnn = []
        for _ in range(num_hidden):
            self.fnn += [nn.Linear(dim, dim), nn.ReLU()]
        self.fnn += [nn.Linear(dim, dim)]

        self.fnn = nn.Sequential(*self.fnn)

    def forward(self, x):
        return self.fnn(x)

"""## init poisson flow"""

poisson = PoissonField(129).to(device)

"""## Train poisson flow"""

LARGE_BATCH_SIZE = 1024
SMALL_BATCH_SIZE = 128
HIDDEN_SIZE = 128
M = 20
GAMMA = 0.3
SIGMA = 0.01
TAU = 0.03

NUM_EPOCHS = 200

after_alexnet_train_datalaoder = DataLoader(after_alexnet_train, batch_size=LARGE_BATCH_SIZE)
after_alexnet_test_datalaoder = DataLoader(after_alexnet_test, batch_size=LARGE_BATCH_SIZE)

optimizer = optim.Adam(poisson.parameters())
criterion = get_loss_fn()

losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
  for large_batch, _ in after_alexnet_train_datalaoder:
      large_batch = large_batch.to(device)
      print(large_batch.shape)
      break
      optimizer.zero_grad()
      loss = criterion(poisson, large_batch)
      loss.backward()
      optimizer.step()

      losses.append(loss.detach().cpu().item())

plt.plot(losses)

torch.save(poisson.state_dict(), "poisson-supv-cifar10.pth")
!cp "poisson-supv-cifar10.pth" "./drive/MyDrive/cifar-10/"

"""## Or load from model.pth"""

!gdown 1-BMh8yFcDf1IzxYo3ZrNK2IuV6rJntP0
poisson.load_state_dict(torch.load("poisson-supv-cifar10.pth"))

"""## Pass through poisson flow"""

def create_ode_forward(delta, steps):
    def ode_forward(samples, v_b):
        with torch.no_grad():
          for _ in (range(steps)):
              field = v_b(samples)
              normalizer = torch.sum(field * samples, dim=-1, keepdims=True)
              step = delta * field / normalizer
              samples += step
        return samples
    return ode_forward

def poisson_pass(poisson, dataloader):
  poisson.eval()

  ode_forward = create_ode_forward(0.01, 100)

  final_reprs = []
  final_labels = []

  for large_batch, labels in tqdm(dataloader):
      LB, _ = large_batch.shape
      large_batch = large_batch.to(device)
      large_batch = torch.cat([large_batch, torch.zeros(LB, 1).to(device)], dim=-1)
      repr = ode_forward(large_batch, poisson)
      final_reprs.append(repr.cpu().numpy())
      final_labels.append(labels.numpy())

  return {"data": torch.tensor(np.vstack(final_reprs)), "labels": torch.tensor(np.hstack(final_labels))}

after_poisson_train = poisson_pass(poisson, after_alexnet_train_datalaoder)
after_poisson_test = poisson_pass(poisson, after_alexnet_test_datalaoder)

from sklearn.linear_model import LogisticRegression

final = LogisticRegression().fit(
    after_poisson_train["data"],
    after_poisson_train["labels"]
)

print("train accuracy:", final.score(
    after_poisson_train["data"],
    after_poisson_train["labels"]
))

print("test accuracy:", final.score(
    after_poisson_test["data"],
    after_poisson_test["labels"]
))
