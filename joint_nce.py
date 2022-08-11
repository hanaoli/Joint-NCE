import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
from torchvision.utils import save_image

def plot_energy(ET, EF):
    x = range(len(ET))
    plt.figure()
    plt.plot(x, ET, color='red', label='Data')
    plt.plot(x, EF, color='blue', label='Noise')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/energy.png")
    plt.close()


def plot_acc(true_acc, false_acc):
    x = range(len(true_acc))
    plt.figure()
    plt.plot(x, true_acc, color='red', label='True Accuracy')
    plt.plot(x, false_acc, color='blue', label='False Accuracy')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/accuracy.png")
    plt.close()

def plot_loss(loss):
    x = range(len(loss))
    plt.figure()
    plt.plot(x, loss, color='red', label='Loss')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/loss.png")
    plt.close()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, latent_dim)
        )
        self.fc1 = nn.Linear(img_dim * img_dim, 1024)
        self.lrelu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc21 = nn.Linear(1024, latent_dim)
        self.fc22 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        o1 = self.lrelu(self.fc1(x))
        o2 = self.lrelu(self.bn(self.fc2(o1)))
        return self.fc21(o2), self.fc22(o2)

'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        return self.layers(x)
'''

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, img_dim * img_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.layers(z)


class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()
        self.c = nn.Parameter(torch.tensor(0).float())
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim + latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, X, z):
        Xz = torch.cat([X, z], dim=1)
        return self.layers(Xz) - self.c

class EnergyX(nn.Module):
    def __init__(self):
        super(EnergyX, self).__init__()
        self.c = nn.Parameter(torch.tensor(0).float())
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.layers(x) - self.c

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

torch.pi = torch.as_tensor(np.pi)
def EBM_loss(image, image_test, z_true, z_fake, G, EBM,  eps=1e-6):
    PT = (1 / (torch.sqrt(2 * torch.pi)) * torch.exp((-(z_true) ** 2) / (2))).sum(dim=1) * 1 / (
                torch.sqrt(2 * torch.pi) * 0.3) * torch.exp(-1 / (2 * 0.3 * 0.3) * (-G(z_true) + image) ** 2).sum(dim=1)
    PF = 1 / (torch.sqrt(2 * torch.pi)) * torch.exp(-(z_fake ** 2) / 2).sum(dim=1) * 1 / (
            torch.sqrt(2 * torch.pi) * 0.3) * torch.exp(-1 / (2 * 0.3 * 0.3) * (-G(z_fake) + image_test) ** 2).sum(
        dim=1)
    ET = torch.exp(EBM(image, z_true).squeeze())
    EF = torch.exp(EBM(G(z_fake), z).squeeze())
    print("PT",PT.mean())
    print("PF", PF.mean())
    print("ET",ET.mean())
    print("EF",EF.mean())
    loss = torch.log(ET/(ET+PT) + eps) + torch.log(PF/(EF+PF) + eps)
    return -torch.mean(loss), ET/(ET+PT), PF/(EF+PF), ET, EF

def EBMX_loss(image, image_test, z_true, z_fake, G, EBM, mu_true, s, mu_fake, s_fake, eps=1e-6):
    PT = 1 / (torch.sqrt(2 * torch.pi)) * torch.exp(-(z_true ** 2) / 2).sum(dim=1) * 1 / (
            torch.sqrt(2 * torch.pi) * 0.3) * torch.exp(-1 / (2 * 0.3 * 0.3) * (-G(z_true) + image) ** 2).sum(dim=1)
    PF = 1 / (torch.sqrt(2 * torch.pi)) * torch.exp(-(z_fake ** 2) / 2).sum(dim=1) * 1 / (
            torch.sqrt(2 * torch.pi) * 0.3) * torch.exp(-1 / (2 * 0.3 * 0.3) * (-G(z_fake) + image_test) ** 2).sum(
        dim=1)

    ET = torch.exp(EBM(image).squeeze()) * (1 / (
            torch.sqrt(2 * torch.pi) * s) * torch.exp(-1 / (2 * s * s) * (-z_true + mu_true) ** 2)).sum(
        dim=1)
    EF = torch.exp(EBM(G(z_fake)).squeeze()) * (1 / (
            torch.sqrt(2 * torch.pi) * s_fake) * torch.exp(-1 / (2 * s_fake * s_fake) * (-z_fake + mu_fake) ** 2)).sum(
        dim=1)
    print("PT", PT.mean())
    print("PF", PF.mean())
    print("ET", ET.mean())
    print("EF", EF.mean())
    loss = torch.log(ET / (ET + PT) + eps) + torch.log(PF / (EF + PF) + eps)
    return -torch.mean(loss), ET / (ET + PT), PF / (EF + PF), ET, EF

folder = "results23_normal_constant_g3lr_1e5_infer"
if not os.path.exists(folder):
    os.makedirs(folder)


latent_dim = 50
img_dim = 28
batch_size = 128
n_epochs = 2000
l_rate = 1e-6
load_model = True




transform = transforms.Compose([transforms.ToTensor()])
train_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True,
                                           transform=transform),
                batch_size=batch_size, shuffle=False)



E = Encoder().cuda()
G = Generator().cuda()
EBM = EnergyX().cuda()

if load_model:
    E = torch.load("vae/e.pth")
    G = torch.load("vae/g.pth")
else:
    E.apply(init_weights)
    G.apply(init_weights)

EBM.apply(init_weights)

#optimizers
optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                lr=0.00001, betas=(0.5, 0.999))
optimizer_EBM = torch.optim.Adam(EBM.parameters(),
                               lr=0.00001, betas=(0.5, 0.999))

ET_list = []
EF_list = []
EBM_loss_list = []
true_acc_list = []
false_acc_list = []

for epoch in range(n_epochs):
    start_time = time.time()
    EBM_loss_acc = 0.
    EG_loss_acc = 0.
    ET_loss = 0.
    EF_loss = 0.
    true_acc_loss = 0.
    false_acc_loss = 0.
    EBM.train()
    E.train()
    G.train()

    for i, (images, labels) in enumerate(train_data):
        # Sample True
        images = images.cuda()
        images = images.reshape(images.size(0), -1)

        # Sample Fake
        z = torch.randn(images.size(0), latent_dim).cuda()

        # Obtain Latent
        mu, logvar = E(images)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).cuda()
        #     true_acc = torch.tensor(0).float()
        #  while true_acc.mean() < 0.5:
        mu, logvar = E(images)
        std = torch.exp(0.5 * logvar)
        z_infer = eps.mul(std).add_(mu)
        print("Zmax", z_infer.max())
        print("Zmin", z_infer.min())
        print("Zmean", z_infer.mean())
        print("Zstd", z_infer.std())
  #      z_infer = E(images)
        x_gen_train = G(z_infer)
        x_gen_test = G(z)
        mu_fake, logvar_fake = E(x_gen_test)
        s_fake = torch.exp(0.5 * logvar_fake)
        # compute losses
    #    loss_EBM, true_acc, false_acc, ET, EF = EBM_loss(images, x_gen_test, z_infer, z, G, EBM)
        loss_EBM, true_acc, false_acc, ET, EF = EBMX_loss(images, x_gen_test, z_infer, z, G, EBM, mu, std, mu_fake, s_fake)
        optimizer_EBM.zero_grad()
        loss_EBM.backward()
        optimizer_EBM.step()

        # Upadte EG
  #      false_acc_g = torch.tensor(0).float()
 #       while false_acc_g.mean() < 0.5:
        mu, logvar = E(images)
        std = torch.exp(0.5 * logvar)
        z_infer = eps.mul(std).add_(mu)
 #       z_infer = E(images)
        x_gen_train = G(z_infer)
        x_gen_test = G(z)
        mu_fake, logvar_fake = E(x_gen_test)
        s_fake = torch.exp(0.5 * logvar_fake)
        # compute Pi
     #   loss_EG, true_acc_g, false_acc_g, ET, EF = EBM_loss(images, x_gen_test, z_infer,  z, G, EBM, mu, std)
        loss_EG, true_acc_g, false_acc_g, ET, EF = EBMX_loss(images, x_gen_test, z_infer,  z, G, EBM, mu, std, mu_fake, s_fake)
        loss_EG = loss_EG * -1
        optimizer_EG.zero_grad()
        loss_EG.backward()
        optimizer_EG.step()


        ET_loss += ET.mean()
        EF_loss += EF.mean()
        EBM_loss_acc += loss_EBM.item()
        EG_loss_acc += loss_EG.item()
        true_acc_loss += true_acc.mean()
        false_acc_loss += false_acc.mean()




    end_time = time.time()
    print('Epoch [{}/{}], Avg_Loss_EBM: {:.4f}, Avg_Loss_EG: {:.4f}, ET_Loss: {:.4f}, EF_Loss: {:.4f}, Time: {:.4f}'
          .format(epoch + 1, n_epochs, EBM_loss_acc / (i+1), EG_loss_acc / (i+1), ET_loss / (i+1), EF_loss / (i+1),
                  end_time - start_time))
    EBM_loss_list.append(EBM_loss_acc)
    ET_list.append(ET_loss.detach().cpu().numpy())
    EF_list.append(EF_loss.detach().cpu().numpy())
    true_acc_list.append(true_acc_loss.detach().cpu().numpy() / (i+1))
    false_acc_list.append(false_acc_loss.detach().cpu().numpy() / (i+1))

    plot_energy(ET_list, EF_list)
    plot_acc(true_acc_list, false_acc_list)
    plot_loss(EBM_loss_list)
    if (epoch) % 5 == 0:

        n_show = 10
        EBM.eval()
        E.eval()
        G.eval()

        with torch.no_grad():

            real = images[:n_show]
            z = torch.rand(n_show, latent_dim).cuda()
            gener = G(z).reshape(n_show, img_dim, img_dim).cpu().numpy()
            mu, logvar = E(real)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).cuda()
            z_infer = eps.mul(std).add_(mu)
       #     z_infer = E(real)
            recon = G(z_infer).reshape(n_show, img_dim, img_dim).cpu().numpy()
            real = real.reshape(n_show, img_dim, img_dim).cpu().numpy()

            fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.05, hspace=0)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('Epoch {}'.format(epoch + 1))
            fig.text(0.04, 0.75, 'G(z)', ha='left')
            fig.text(0.04, 0.5, 'x', ha='left')
            fig.text(0.04, 0.25, 'G(E(x))', ha='left')

            for i in range(n_show):
                ax[0, i].imshow(gener[i], cmap='gray')
                ax[0, i].axis('off')
                ax[1, i].imshow(real[i], cmap='gray')
                ax[1, i].axis('off')
                ax[2, i].imshow(recon[i], cmap='gray')
                ax[2, i].axis('off')
            plt.savefig(folder + "/plot_energy_%05d.png" % (epoch))
            plt.close()

            z = torch.rand(64, latent_dim).cuda()
            gener = G(z).reshape(64, 1, img_dim, img_dim).cpu()
            save_image(gener, folder + "/generation_%05d.png" % (epoch))
