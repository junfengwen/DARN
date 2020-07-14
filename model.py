import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import L2ProjFunction, GradientReversalLayer
import utils


########## Some components ##########
class MLPNet(nn.Module):

    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        if self.process_final:
            return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return self.final(x)


class ConvNet(nn.Module):

    def __init__(self, configs):
        """
        Feature extractor for the image (digits) datasets
        """

        super().__init__()
        self.channels = configs["channels"]  # number of channels
        self.num_conv_layers = len(configs["conv_layers"])
        self.num_channels = [self.channels] + configs["conv_layers"]
        # Parameters of hidden, cpcpcp, feature learning component.
        self.convs = nn.ModuleList([nn.Conv2d(self.num_channels[i],
                                              self.num_channels[i+1],
                                              kernel_size=3)
                                    for i in range(self.num_conv_layers)])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability

    def forward(self, x):

        dropout = self.dropout
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(dropout(x))), 2, 2, ceil_mode=True)
        x = x.view(x.size(0), -1)  # flatten
        return x


########## Models ##########
# DARN and MDAN
class DarnBase(nn.Module):

    def __init__(self, configs):
        """
        Domain AggRegation Network.
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        self.mode = mode = configs["mode"]
        self.mu = configs["mu"]
        self.gamma = configs["gamma"]

        if mode == "L2":
            self.proj = L2ProjFunction.apply
        else:
            self.proj = None

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []
        for i in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[i]))
        t_features = self.feature_net(tinputs)

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_src_domains):
            logprobs.append(F.log_softmax(self.class_net(s_features[i]), dim=1))
        train_losses = torch.stack([F.nll_loss(logprobs[i], soutputs[i])
                                    for i in range(self.num_src_domains)])

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_src_domains):
            sdomains.append(self.domain_nets[i](
                self.grl(s_features[i])))
            tdomains.append(self.domain_nets[i](
                self.grl(t_features)))

        batch_size = tinputs.shape[0]
        slabels = torch.ones([batch_size, 1], requires_grad=False,
                             dtype=torch.float32, device=tinputs.device)
        tlabels = torch.zeros([batch_size, 1], requires_grad=False,
                              dtype=torch.float32, device=tinputs.device)
        domain_losses = torch.stack([F.binary_cross_entropy_with_logits(sdomains[i], slabels) +
                                     F.binary_cross_entropy_with_logits(tdomains[i], tlabels)
                                     for i in range(self.num_src_domains)])

        return self._aggregation(train_losses, domain_losses)

    def _aggregation(self, train_losses, domain_losses):
        """
        Aggregate the losses into a scalar
        """

        mu, alpha = self.mu, None
        if self.num_src_domains == 1:  # dann
            loss = train_losses + mu * domain_losses
        else:
            mode, gamma = self.mode, self.gamma
            if mode == "dynamic":  # mdan
                g = (train_losses + mu * domain_losses) * gamma
                loss = torch.logsumexp(g, dim=0) / gamma
            elif mode == "L2":  # darn
                g = gamma * (train_losses + mu * domain_losses)
                alpha = self.proj(g)
                loss = torch.dot(g, alpha) + torch.norm(alpha)
                alpha = alpha.cpu().detach().numpy()
            else:
                raise NotImplementedError("Unknown aggregation mode %s" % mode)

        return loss, alpha

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class DarnMLP(DarnBase):

    def __init__(self, configs):
        """
        DARN with MLP
        """

        super().__init__(configs)

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}
        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])

        self.domain_nets = nn.ModuleList([nn.Linear(configs["hidden_layers"][-1], 1)
                                          for _ in range(self.num_src_domains)])


class DarnConv(DarnBase):

    def __init__(self, configs):
        """
        DARN with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.class_net = MLPNet(cls_configs)

        dom_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["dom_fc_layers"],
                       "output_dim": 1,
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.domain_nets = nn.ModuleList([MLPNet(dom_configs)
                                          for _ in range(self.num_src_domains)])


# MDMN
class MdmnBase(nn.Module):

    def __init__(self, configs):
        """
        MDMN model
        """

        super().__init__()

        self.num_src_domains = configs["num_src_domains"]
        self.num_domains = configs["num_src_domains"] + 1
        self.mu = configs["mu"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []
        for i in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[i]))
        t_features = self.feature_net(tinputs)

        # These will be used later to compute the gradient penalty
        src_rand = s_features[int(np.random.choice(self.num_src_domains, 1))]
        epsilon = np.random.rand()
        interpolated = epsilon * src_rand + (1 - epsilon) * t_features

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_src_domains):
            logprobs.append(F.log_softmax(self.class_net(s_features[i]), dim=1))
        train_losses = torch.stack([F.nll_loss(logprobs[i], soutputs[i])
                                    for i in range(self.num_src_domains)])

        # Domain classification accuracies.
        pred = [self.domain_net(self.grl(s_features[i]))
                for i in range(self.num_src_domains)]
        pred.append(self.domain_net(self.grl(t_features)))
        pred = torch.cat(pred, dim=0)  # pred are the logits of the domain prediction
        inter_f = self.domain_net(interpolated)

        batch_size = tinputs.shape[0]
        d_idx = np.concatenate([np.ones(batch_size, dtype=np.int) * i
                                for i in range(self.num_domains)])
        # convert to one-hot
        d = np.zeros((d_idx.size, self.num_domains), dtype=np.float32)
        d[np.arange(d_idx.size), d_idx] = 1
        # compute weights
        weights, alpha = utils.compute_weights(d, pred.cpu().detach().numpy(), batch_size)
        weights = torch.from_numpy(weights).float().to(pred.device).detach()

        # The following compute the penalty of the Lipschitz constant
        penalty_coefficient = 10.
        # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
        # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
        f_gradient = torch.autograd.grad(torch.sum(inter_f), interpolated,
                                         create_graph=True, retain_graph=True)[0]
        f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
        f_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - 1.0) ** 2)
        domain_losses = torch.mean(weights * pred) + f_gradient_penalty

        loss = torch.mean(train_losses) + self.mu * domain_losses

        return loss, alpha

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class MdmnMLP(MdmnBase):

    def __init__(self, configs):
        """
        MDMN with MLP
        """

        super().__init__(configs)

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}

        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])

        self.domain_net = nn.Linear(configs["hidden_layers"][-1],
                                    self.num_domains)


class MdmnConv(MdmnBase):

    def __init__(self, configs):
        """
        MDMN with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.class_net = MLPNet(cls_configs)

        dom_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["dom_fc_layers"],
                       "output_dim": self.num_domains,
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.domain_net = MLPNet(dom_configs)


# MSDA
class MsdaBase(nn.Module):

    def __init__(self):
        """
        Moment matching for multi-source domain adaptation. ICCV 2019
        """

        super().__init__()
        self.num_src_domains = None
        self.feature_net = None
        self.class_net1 = None
        self.class_net2 = None
        self.mu = None

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(training loss, discrepancy loss)
        """

        # Compute features
        s_features = []
        for i in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[i]))
        t_features = self.feature_net(tinputs)

        # Loss and predictions
        train_loss1, train_loss2 = 0., 0.
        for i in range(self.num_src_domains):
            logprobs1 = F.log_softmax(self.class_net1(s_features[i]), dim=1)
            train_loss1 = train_loss1 + torch.mean(F.nll_loss(logprobs1, soutputs[i]))
            logprobs2 = F.log_softmax(self.class_net2(s_features[i]), dim=1)
            train_loss2 = train_loss2 + torch.mean(F.nll_loss(logprobs2, soutputs[i]))

        t_pred1 = F.softmax(self.class_net1(t_features), dim=1)
        t_pred2 = F.softmax(self.class_net2(t_features), dim=1)

        # Combined by MSDA
        # https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/be3aef641719c0020c8bf11d8ed8b9df79736f6a/M3SDA/code_MSDA_digit/solver_MSDA.py#L286
        loss_msda = utils.msda_regulizer(s_features, t_features)
        train_loss = train_loss1 + train_loss2 + self.mu * loss_msda

        disc_loss = torch.mean(torch.abs(t_pred1 - t_pred2))

        return train_loss, disc_loss

    def inference(self, x):

        x = self.feature_net(x)
        # Classification probability.
        return F.log_softmax(self.class_net1(x), dim=1)


class MsdaMLP(MsdaBase):

    def __init__(self, configs):
        """
        MSDA with MLP
        """

        super().__init__()
        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}
        self.feature_net = MLPNet(fea_configs)
        self.G_params = self.feature_net.parameters()

        self.num_src_domains = configs["num_src_domains"]
        # Parameter of the final softmax classification layer.
        self.class_net1 = nn.Linear(configs["hidden_layers"][-1],
                                    configs["num_classes"])
        self.class_net2 = nn.Linear(configs["hidden_layers"][-1],
                                    configs["num_classes"])
        self.C1_params = self.class_net1.parameters()
        self.C2_params = self.class_net2.parameters()
        self.mu = 5e-4


class MsdaConv(MsdaBase):

    def __init__(self, configs):
        """
        MSDA with convolution feature extractor
        """

        super().__init__()
        self.feature_net = ConvNet(configs)
        self.G_params = self.feature_net.parameters()

        self.num_src_domains = configs["num_src_domains"]
        # Parameter of the final softmax classification layer.
        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.class_net1 = MLPNet(cls_configs)
        self.class_net2 = MLPNet(cls_configs)
        self.C1_params = self.class_net1.parameters()
        self.C2_params = self.class_net2.parameters()
        self.mu = 1e-7
