import os
import numpy as np
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from sksurv.metrics import concordance_index_censored
from thop import profile, clever_format
import torch.optim
import torch.nn.parallel
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from hypll.optim import RiemannianAdam
import datetime
from lifelines.statistics import logrank_test


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def plot_loss_index(training_losses, training_indices, validation_losses, validation_indices, results_directory, fold_number):
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    epoch_count = len(training_losses)
    epochs = list(range(1, epoch_count + 1))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_losses, label='Train Loss', color='blue')
    plt.ylim(1, 9)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epoch')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_indices, label='Train Index', color='orange')
    plt.ylim(0.4, 0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Train Index')
    plt.title('Train Index vs Epoch')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, validation_losses, label='Validation Loss', color='green')
    plt.ylim(1, 9)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, validation_indices, label='Validation Index', color='red')
    plt.ylim(0.4, 0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Index')
    plt.title('Validation Index vs Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, f'loss_index_per_epoch__{fold_number}.png'))
    plt.close()

    combined_fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(epochs, training_losses, label='Train Loss', color='blue', linestyle='--')
    ax1.plot(epochs, validation_losses, label='Validation Loss', color='green', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(1, 9)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Index', color='red')
    ax2.plot(epochs, training_indices, label='Train Index', color='orange')
    ax2.plot(epochs, validation_indices, label='Validation Index', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0.4, 0.8)

    combined_fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Loss and Index vs Epoch')
    plt.savefig(os.path.join(results_directory, f'combined_loss_index__{fold_number}.png'))
    plt.close()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(**inputs)


class Engine(object):
    def __init__(self, config, results_directory, fold_number):
        self.config = config
        self.results_directory = results_directory
        self.fold_number = fold_number

        if config.log_data:
            from tensorboardX import SummaryWriter
            writer_directory = os.path.join(results_directory, 'fold_' + str(fold_number))
            os.makedirs(writer_directory, exist_ok=True)
            self.writer = SummaryWriter(writer_directory, flush_secs=15)

        self.best_score = 0
        self.best_epoch = 0
        self.best_filename = None
        self.execution_time = None

    def learning(self, current_time, model, train_loader, val_loader, loss_function, optimizer, scheduler, dataset):
        self.execution_time = current_time

        if torch.cuda.is_available():
            model = model.cuda()

        if self.config.resume is not None:
            if os.path.isfile(self.config.resume):
                print("=> loading checkpoint '{}'".format(self.config.resume))
                checkpoint = torch.load(self.config.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.config.resume))

        if self.config.evaluate:
            self.validate(val_loader, model, loss_function, self.config.modality)
            return

        train_losses = []
        train_indices = []
        val_losses = []
        val_indices = []

        for epoch in range(self.config.num_epoch):
            self.epoch = epoch
            train_loss, train_index = self.train(train_loader, model, loss_function, optimizer, epoch, dataset)

            train_losses.append(train_loss)
            train_indices.append(train_index)

            c_index, val_loss, val_index = self.validate(val_loader, model, loss_function, self.config.modality, epoch, dataset)
            val_losses.append(val_loss)
            val_indices.append(val_index)

            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score
                })

            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))

            if scheduler is not None:
                scheduler.step()

            print('>')

        plot_loss_index(train_losses, train_indices, val_losses, val_indices, self.results_directory, self.fold_number)
        return self.best_score, self.best_epoch

    def train(self, data_loader, model, loss_function, optimizer, epoch, dataset):
        model.train()
        total_loss = 0.0
        risk_scores = np.zeros((len(data_loader)))
        censorships = np.zeros((len(data_loader)))
        event_times = np.zeros((len(data_loader)))
        total_flops, total_params = 0.0, 0.0
        progress_bar = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))

        for batch_index, (data_wsi, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, labels, event_times_batch, censorships_batch) in enumerate(progress_bar):

            if torch.cuda.is_available():
                data_wsi = data_wsi.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                labels = labels.type(torch.LongTensor).cuda()
                censorships_batch = censorships_batch.type(torch.FloatTensor).cuda()

            hazards, survival_prob, predictions, predicted_probs, group_predictions, group_predicted_probs, m_loss = model(
                x_path=data_wsi, x_omic1=data_omic1, x_omic2=data_omic2,
                x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                x_omic6=data_omic6
            )

            survival_loss = loss_function[0](hazards=hazards, S=survival_prob, Y=labels, c=censorships_batch)
            similarity_loss_P = loss_function[1](predictions.detach(), predicted_probs)
            similarity_loss_G = loss_function[1](group_predictions.detach(), group_predicted_probs)
            loss = survival_loss + self.config.alpha * (similarity_loss_P + similarity_loss_G)

            if self.config.MoELoss:
                loss += self.config.LossRate * m_loss

            risk = -torch.sum(survival_prob, dim=1).detach().cpu().numpy()
            risk_scores[batch_index] = risk
            censorships[batch_index] = censorships_batch.item()
            event_times[batch_index] = event_times_batch
            total_loss += loss.item()
            euclidean_params = [p for name, p in model.named_parameters() if 'hyperbolic' not in name]

            optimizer.param_groups.clear()

 
            optimizer.add_param_group({'params': euclidean_params})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index == 0 and epoch == 0:
                input_data = {
                    "x_path": data_wsi,
                    "x_omic1": data_omic1,
                    "x_omic2": data_omic2,
                    "x_omic3": data_omic3,
                    "x_omic4": data_omic4,
                    "x_omic5": data_omic5,
                    "x_omic6": data_omic6,
                }
                total_flops += FlopCountAnalysis(model, input_data).flops
                total_params += parameter_count_table(model)

        mean_loss = total_loss / len(data_loader)
        c_index = concordance_index_censored(censorships, event_times, risk_scores)

        return mean_loss, c_index

    def validate(self, data_loader, model, loss_function, modality, epoch, dataset):
        model.eval()
        total_loss = 0.0
        risk_scores = np.zeros((len(data_loader)))
        censorships = np.zeros((len(data_loader)))
        event_times = np.zeros((len(data_loader)))
        progress_bar = tqdm(data_loader, desc='Validate Epoch: {}'.format(epoch))

        with torch.no_grad():
            for batch_index, (data_wsi, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, labels, event_times_batch, censorships_batch) in enumerate(progress_bar):
                if torch.cuda.is_available():
                    data_wsi = data_wsi.cuda()
                    data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                    data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                    data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                    data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                    data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                    data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                    labels = labels.type(torch.LongTensor).cuda()
                    censorships_batch = censorships_batch.type(torch.FloatTensor).cuda()

                hazards, survival_prob, predictions, predicted_probs, group_predictions, group_predicted_probs, m_loss = model(
                    x_path=data_wsi, x_omic1=data_omic1, x_omic2=data_omic2,
                    x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                    x_omic6=data_omic6
                )

                survival_loss = loss_function[0](hazards=hazards, S=survival_prob, Y=labels, c=censorships_batch)


                similarity_loss_P = loss_function[1](predictions.detach(), predicted_probs)
                similarity_loss_G = loss_function[1](group_predictions.detach(), group_predicted_probs)
                loss = survival_loss + self.config.alpha * (similarity_loss_P + similarity_loss_G)


                risk = -torch.sum(survival_prob, dim=1).detach().cpu().numpy()
                risk_scores[batch_index] = risk
                censorships[batch_index] = censorships_batch.item()
                event_times[batch_index] = event_times_batch
                total_loss += loss.item()


        mean_loss = total_loss / len(data_loader)
        c_index = concordance_index_censored(censorships, event_times, risk_scores)

        return c_index, mean_loss, c_index

    def save_checkpoint(self, state):
        filename = os.path.join(self.results_directory, 'checkpoint.pth')
        torch.save(state, filename)
