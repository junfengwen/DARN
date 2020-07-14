# This file define custom functions with forward and backward passes.
# See https://pytorch.org/docs/stable/notes/extending.html for more details.
# Also https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
import torch


class L2ProjFunction(torch.autograd.Function):
    """
    This function defines the L2 projection for the input z.
    The forward pass uses a binary search and saves some quantities for the backward pass.
    The backward pass computes the Jacobian-vector product as in the Appendix of the paper.
    Note: if the forward pass loops forever, you may relax the termination condition a little bit.
    """

    @staticmethod
    def forward(self, z, dim=-1):

        z = z.transpose(dim, -1)
        left = torch.min(z, dim=-1, keepdim=True)[0]
        right = torch.max(z, dim=-1, keepdim=True)[0] + 1.0
        alpha_norm = torch.tensor(100.0, dtype=torch.float, device=z.device)
        one = torch.tensor(1.0, dtype=torch.float, device=z.device)
        # zero = torch.tensor(0.0, dtype=torch.float, device=z.device)
        # while not torch.allclose(right - left, zero):
        while not torch.allclose(alpha_norm, one):
            mid = left + (right - left) * 0.5
            alpha = torch.relu(mid - z)
            alpha_norm = torch.norm(alpha, dim=-1, keepdim=True)
            right[alpha_norm > 1.0] = mid[alpha_norm > 1.0]
            left[alpha_norm <= 1.0] = mid[alpha_norm <= 1.0]
        K = alpha.sum(-1, keepdim=True)
        alpha = alpha / K
        s = (alpha > 0).float()  # support, positivity mask
        zs = z * s
        S = s.sum(-1, keepdim=True)
        A = zs.sum(-1, keepdim=True) ** 2 - S * ((zs ** 2).sum(-1, keepdim=True) - 1)  # should have A > 0
        self.save_for_backward(alpha, K, s, S, A, torch.tensor(dim))
        return alpha.transpose(dim, -1)

    @staticmethod
    def backward(self, grad_output):

        alpha, K, s, S, A, dim = self.saved_tensors
        dim = dim.item()
        grad_output = grad_output.transpose(dim, -1)
        # first part
        vhat = (s * grad_output).sum(-1, keepdim=True) / S
        grad1 = (s / K) * (vhat - grad_output)
        # second part
        alpha_s = alpha * s - s / S
        grad2 = S / A.sqrt() * alpha_s * (alpha_s * grad_output).sum(-1, keepdim=True)

        return (grad1 - grad2).transpose(dim, -1)


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(self, inputs):

        return inputs

    @staticmethod
    def backward(self, grad_output):

        grad_input = -grad_output.clone()
        return grad_input

