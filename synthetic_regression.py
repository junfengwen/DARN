# This script generates the figures of the synthetic regression problem
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import torch.optim as optim

from module import L2ProjFunction

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class MiniDARN(nn.Module):

    def __init__(self):
        """
        Simplified DARN
        """

        super().__init__()
        self.regress_net = nn.Linear(1, 1)
        self.gamma = 1.
        self.mu = 0.01
        self.proj = L2ProjFunction.apply

    def forward(self, s_inputs, s_outputs, t_input):

        num_src_domains = len(s_inputs)
        train_losses, discs = [], []
        Phi_t = torch.cat([t_input, torch.ones_like(t_input)], dim=1)
        M_t = torch.mm(Phi_t.transpose(0, 1), Phi_t)

        for i in range(num_src_domains):

            prediction = self.regress_net(s_inputs[i])
            train_loss = torch.mean((prediction - s_outputs[i]) ** 2)
            train_losses.append(train_loss)

            Phi_s = torch.cat([s_inputs[i], torch.ones_like(s_inputs[i])], dim=1)
            M_s = torch.mm(Phi_s.transpose(0, 1), Phi_s)
            disc = torch.max(torch.norm(torch.eig(M_t - M_s)[0], dim=1))
            discs.append(disc)

        train_losses = torch.stack(train_losses)
        discs = torch.stack(discs)

        g = self.gamma * (train_losses + self.mu * discs)
        alpha = self.proj(g)
        loss = torch.dot(g, alpha) + torch.norm(alpha)
        alpha = alpha.detach().cpu().numpy()

        return loss, alpha

    def inference(self, x):

        return self.regress_net(x)


def main():

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    to_plot = True
    figsize = (7, 4)
    num_train = 8
    num_samples = 100

    legend_columnspacing = 1.0

    x_grid = np.arange(num_samples + 1)
    x_grid = x_grid * 2. * np.pi / num_samples
    x_grid -= np.pi

    if to_plot:
        plt.figure(figsize=figsize)
        plt.plot(x_grid, np.sin(x_grid), 'k-')
        font = {'size': 14,
                'family': 'serif'}
        rc('font', **font)

    # Training data
    train_gap = 2 * np.pi / num_train
    train_sig = 0.2
    noise = 0.05

    train_x = []
    train_y = []

    cmap = plt.get_cmap('jet')
    train_colors = cmap(np.linspace(0, 1.0, num_train))

    for i in range(num_train):

        train_mu = train_gap * (0.5 + i) - np.pi
        x_data = np.random.randn(num_samples) * train_sig + train_mu
        train_x.append(x_data)
        y_data = np.sin(x_data) + np.random.randn(num_samples) * noise
        train_y.append(y_data)
        if to_plot:
            plt.plot(x_data, y_data, '.', color=train_colors[i],
                     label="Source %d" % (i+1))

    # Plot training
    if to_plot:
        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.15),
                   fancybox=True,
                   shadow=True,
                   columnspacing=legend_columnspacing,
                   ncol=4)
        plt.tight_layout()
        plt.ylim(-1.2, 1.2)
        plt.savefig("regression_train.pdf")

    # Test data
    num_test = 4
    test_gap = 2 * np.pi / num_test
    test_sig = 0.4
    test_x = []
    test_y = []

    if to_plot:
        plt.figure(figsize=figsize)
        plt.plot(x_grid, np.sin(x_grid), 'k-')

    test_colors = cmap(np.linspace(0, 1.0, num_test))
    for i in range(num_test):

        test_mu = test_gap * (0.5 + i) - np.pi
        x_data = np.random.randn(num_samples) * test_sig + test_mu
        test_x.append(x_data)
        y_data = np.sin(x_data) + np.random.randn(num_samples) * noise
        test_y.append(y_data)
        if to_plot:
            plt.plot(x_data, y_data, 'x', color=test_colors[i],
                     label="Target %d" % (i + 1))

    # PyTorch preprocessing
    s_inputs = [torch.tensor(train_x[i].reshape([num_samples, 1]),
                             requires_grad=False, dtype=torch.float32)
                for i in range(num_train)]
    s_outputs = [torch.tensor(train_y[i].reshape([num_samples, 1]),
                              requires_grad=False, dtype=torch.float32)
                for i in range(num_train)]
    max_iter = 1000
    log_iter = 100
    plot_gap = 0.1

    alpha_all = np.zeros([num_test, num_train, max_iter])
    for t in range(num_test):

        t_input = torch.tensor(test_x[t].reshape([num_samples, 1]),
                               requires_grad=False, dtype=torch.float32)
        model = MiniDARN()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        print("Running target set %d" % t)
        for i in range(max_iter):
            optimizer.zero_grad()
            loss, alpha = model(s_inputs, s_outputs, t_input)
            loss.backward()
            optimizer.step()
            alpha_all[t, :, i] = alpha
            if i % log_iter == 0:
                print("Loss = %.6f" % loss.detach().cpu().numpy())

        # plot
        x_grid = np.arange(np.amin(test_x[t]) - plot_gap,
                           np.amax(test_x[t]) + plot_gap,
                           plot_gap)
        x_grid_tensor = torch.tensor(x_grid.reshape(len(x_grid), 1),
                                     dtype=torch.float32)
        t_predictions = model.inference(x_grid_tensor).detach().numpy()
        if to_plot:
            plt.plot(x_grid, t_predictions, '--', color=test_colors[t],
                     label="Model %d" % (t + 1))

    # Plot test
    if to_plot:
        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.15),
                   fancybox=True,
                   shadow=True,
                   columnspacing=legend_columnspacing,
                   ncol=4)
        plt.tight_layout()
        plt.ylim(-1.2, 1.2)
        plt.savefig("regression_test.pdf")

    # Plot domain weights
    if to_plot:
        x = ["Target %d" % i for i in range(1, num_test + 1)]
        y = alpha_all[:, :, -1]
        plt.figure(figsize=figsize)
        plt.bar(x, y[:, 0], label='Source 1', color=train_colors[0])
        for i in range(1, num_train):
            # stack on top.
            plt.bar(x, y[:, i], color=train_colors[i],
                    bottom=y[:, i - 1], label="Source %d" % (i+1))

        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.15),
                   fancybox=True,
                   shadow=True,
                   columnspacing=legend_columnspacing,
                   ncol=4)
        plt.tight_layout()
        plt.savefig("regression_alpha.pdf")


if __name__ == '__main__':
    main()
