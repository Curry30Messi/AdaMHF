import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def create_loss(args):
    if args.loss == "ce_surv":
        loss_fn = CrossEntropySurvLoss(alpha=0.0)
    elif args.loss == "nll_surv":
        loss_fn = NLLSurvLoss(alpha=0.0)
    elif args.loss == "cox_surv":
        loss_fn = CoxSurvLoss()
    elif args.loss == "nll_surv_kl":
        print('########### ', "nll_surv_kl")
        loss_fn = [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif args.loss == "nll_surv_mse":
        print('########### ', "nll_surv_mse")
        loss_fn = [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif args.loss == "nll_surv_l1":
        print('########### ', "nll_surv_l1")
        loss_fn = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
    elif args.loss == "nll_surv_cos":
        print('########### ', "nll_surv_cos")
        loss_fn = [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif args.loss == "nll_surv_ol":
        print('########### ', "nll_surv_ol")
        loss_fn = [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    else:
        raise NotImplementedError
    return loss_fn


def compute_nll_loss(hazards, survival_probabilities, Y, censoring_status, alpha=0.4, epsilon=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    censoring_status = censoring_status.view(batch_size, 1).float()
    if survival_probabilities is None:
        survival_probabilities = torch.cumprod(1 - hazards, dim=1)
    padded_survival = torch.cat([torch.ones_like(censoring_status), survival_probabilities], 1)
    uncensored_loss = -(1 - censoring_status) * (
        torch.log(torch.gather(padded_survival, 1, Y).clamp(min=epsilon)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=epsilon))
    )
    censored_loss = -censoring_status * torch.log(torch.gather(padded_survival, 1, Y + 1).clamp(min=epsilon))
    neg_log_likelihood = censored_loss + uncensored_loss
    total_loss = (1 - alpha) * neg_log_likelihood + alpha * uncensored_loss
    average_loss = total_loss.mean()
    return average_loss


def compute_ce_loss(hazards, survival_probabilities, Y, censoring_status, alpha=0.4, epsilon=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    censoring_status = censoring_status.view(batch_size, 1).float()
    if survival_probabilities is None:
        survival_probabilities = torch.cumprod(1 - hazards, dim=1)
    padded_survival = torch.cat([torch.ones_like(censoring_status), survival_probabilities], 1)
    regularization = -(1 - censoring_status) * (torch.log(torch.gather(padded_survival, 1, Y) + epsilon) + torch.log(torch.gather(hazards, 1, Y).clamp(min=epsilon)))
    cross_entropy_loss = -censoring_status * torch.log(torch.gather(survival_probabilities, 1, Y).clamp(min=epsilon)) - (1 - censoring_status) * torch.log(1 - torch.gather(survival_probabilities, 1, Y).clamp(min=epsilon))
    total_loss = (1 - alpha) * cross_entropy_loss + alpha * regularization
    average_loss = total_loss.mean()
    return average_loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival_probabilities, Y, censoring_status, alpha=None):
        if alpha is None:
            return compute_ce_loss(hazards, survival_probabilities, Y, censoring_status, alpha=self.alpha)
        return compute_ce_loss(hazards, survival_probabilities, Y, censoring_status, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival_probabilities, Y, censoring_status, alpha=None):
        if alpha is None:
            return compute_nll_loss(hazards, survival_probabilities, Y, censoring_status, alpha=self.alpha)
        return compute_nll_loss(hazards, survival_probabilities, Y, censoring_status, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(self, hazards, survival_times, censoring_status, **kwargs):
        current_batch_size = len(survival_times)
        risk_matrix = np.zeros([current_batch_size, current_batch_size], dtype=int)
        for i in range(current_batch_size):
            for j in range(current_batch_size):
                risk_matrix[i, j] = survival_times[j] >= survival_times[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        risk_matrix_tensor = torch.FloatTensor(risk_matrix).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        cox_loss = -torch.mean((theta - torch.log(torch.sum(exp_theta * risk_matrix_tensor, dim=1))) * (1 - censoring_status))
        return cox_loss


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        positive_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        negative_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )

        loss = positive_pairs + self.gamma * negative_pairs
        return loss
