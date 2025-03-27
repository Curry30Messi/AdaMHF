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
            # hyperbolic_params = [p for name, p in model.named_parameters() if 'hyperbolic' in name]
            #
            # # 定义优化器
            # optimizer_euclidean = torch.optim.SGD(filter(lambda p: p.requires_grad, euclidean_params), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
            # 假设 optimizer 已经定义
            # 首先清空 optimizer 的 param_groups
            optimizer.param_groups.clear()

            # 然后只将 euclidean_params 添加到 optimizer 中
            optimizer.add_param_group({'params': euclidean_params})

            # optimizer_euclidean = torch.optim.Adam(
            #     filter(lambda p: p.requires_grad, euclidean_params),
            #     lr=self.args.lr,
            #     weight_decay=self.args.weight_decay
            # )

            # optimizer_hyperbolic = RiemannianAdam(hyperbolic_params, lr=0.001)
            # =======================================


            # for name, parms in model.named_parameters():
            #     if parms.grad is not None:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #               ' -->grad_value:', torch.mean(parms.grad))
            #     else:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #               ' -->grad_value: None', )
            optimizer.zero_grad()
            # optimizer_hyperbolic.zero_grad()
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
            # combined_features = torch.cat((P_hat, G_hat), dim=1)
            # combined_features=combined_features.unsqueeze(1)
            # # combined_features=self.projector(combined_features)
            # features = self.classifier_model.text_model(inputs_embeds=combined_features).last_hidden_state[:, 0, :]  # 使用[CLS]token
            # features = features.squeeze(1)
            # risk_predictions = self.classification_head(features)
            # risk_predictions = (risk_predictions > 0.5).long()

                similarity_loss_P = loss_function[1](predictions.detach(), predicted_probs)
                similarity_loss_G = loss_function[1](group_predictions.detach(), group_predicted_probs)
                loss = survival_loss + self.config.alpha * (similarity_loss_P + similarity_loss_G)

            # print("risk: ",risk)
            # risk_labels = (risk > 0.5).long()
            # print("risk_labels: ",risk_labels)
            # print("risk_predictions: ",risk_predictions)
            # classification_loss = self.classification_criterion(risk_predictions, risk_labels)
            # print("classification_loss: ",classification_loss)
                risk = -torch.sum(survival_prob, dim=1).detach().cpu().numpy()
                risk_scores[batch_index] = risk
                censorships[batch_index] = censorships_batch.item()
                event_times[batch_index] = event_times_batch
                total_loss += loss.item()

#   for i in range(attn_weights.shape[0]):
#                 row, col = divmod(i, 4)  # 2 行 4 列
#                 ax = fig.add_subplot(spec[row, col])
#                 attn_map = attn_weights[i].detach().cpu().numpy()  # 转为 NumPy 数组
#                 im = ax.imshow(attn_map, cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax)  # 统一颜色范围
#                 ax.set_title(f"Head {i}", fontsize=10)
#                 ax.set_xlabel("Token", fontsize=8)
#                 ax.set_ylabel("Token", fontsize=8)
#                 axes.append(ax)

#             # 添加统一的颜色条
#             cbar_ax = fig.add_subplot(spec[:, 4])  # 在右侧预留一列用于颜色条
#             cbar = fig.colorbar(im, cax=cbar_ax)
#             cbar.set_label("Attention Weight", fontsize=12)
#             output_dir = f'results_heatmap/_{dataset}/_{self.time}_alpha{self.args.alpha}_modality{self.args.modality}_Rate{self.args.Rate}_epoch{self.args.num_epoch}/test'
#             os.makedirs(output_dir, exist_ok=True)
#             output_path = os.path.join(output_dir,  f"__{self.fold}__.png")
#             plt.savefig(output_path, dpi=600)
#             plt.close(fig)


#             # survival loss + sim loss + sim loss
#             sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
#             sim_loss_P = criterion[1](P.detach(), P_hat)
#             sim_loss_G = criterion[1](G.detach(), G_hat)
#             loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)
#             if self.args.MoELoss:
#                 loss+=self.args.LossRate*MLoss
#             # print("======================validate==================")
#             # print("loss:",loss)
#             # print("sur_loss:",sur_loss)
#             # print("self.args.alpha * (sim_loss_P + sim_loss_G)",self.args.alpha * (sim_loss_P + sim_loss_G))
#             print("S: ",S)
#             risk = -torch.sum(S, dim=1).cpu().numpy()
#             print("risk: ",risk)
#             all_risk_scores[batch_idx] = risk
#             all_censorships[batch_idx] = c.cpu().numpy()
#             all_event_times[batch_idx] = event_time
#             val_loss += loss.item()

#         if epoch == self.args.num_epoch - 2:
#             plt.clf()
#             # 打印数据长度
#             print("all_censorships", len(all_censorships))
#             print("all_event_times", len(all_event_times))
#             print("all_risk_scores", len(all_risk_scores))

#             # 复制数据以避免修改原始数据
#             all_censorships_temp = all_censorships.copy()
#             all_event_times_temp = all_event_times.copy()
#             all_risk_scores_temp = all_risk_scores.copy()

#             kmf = KaplanMeierFitter()
#             median_risk = np.median(all_risk_scores_temp)

#             low_risk_group = all_risk_scores_temp >= median_risk
#             high_risk_group = all_risk_scores_temp < median_risk

#             # 绘制低风险组生存曲线
#             kmf.fit(all_event_times_temp[low_risk_group], all_censorships_temp[low_risk_group], label="Low Risk")
#             ax = kmf.plot_survival_function()

#             # 绘制高风险组生存曲线
#             kmf.fit(all_event_times_temp[high_risk_group], all_censorships_temp[high_risk_group], label="High Risk")
#             kmf.plot_survival_function(ax=ax)

#             # 使用log-rank test计算p-value
#             results = logrank_test(all_event_times_temp[low_risk_group], all_event_times_temp[high_risk_group],
#                                    event_observed_A=all_censorships_temp[low_risk_group],
#                                    event_observed_B=all_censorships_temp[high_risk_group])

#             p_value_text = f'p-value: {results.p_value:.1e}'
#             plt.text(0.6, 0.2, p_value_text, transform=ax.transAxes, fontsize=16,  # 增大 p-value 字体
#                      bbox=dict(facecolor='white', alpha=0.5))

#             plt.xlabel('Time (months)', fontsize=14)  # 增大 x 轴标签字体
#             plt.ylabel('Overall Survival', fontsize=14)  # 增大 y 轴标签字体

#             # 增加图例并设置字体大小
#             plt.legend(fontsize=12)  # 设置图例字体大小
#             # 保存图像
#             dataset = dataset[4:]
#             output_dir = f'results_img/_{dataset}/_{self.time}_alpha{self.args.alpha}_modality{self.args.modality}_Rate{self.args.Rate}_epoch{self.args.num_epoch}/test'
#             os.makedirs(output_dir, exist_ok=True)
#             output_path = os.path.join(output_dir,  f"__{self.fold}__.png")
#             plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
#             # plt.show()

#             print(f"img saved to: {output_path}")


        mean_loss = total_loss / len(data_loader)
        c_index = concordance_index_censored(censorships, event_times, risk_scores)

        return c_index, mean_loss, c_index

    def save_checkpoint(self, state):
        filename = os.path.join(self.results_directory, 'checkpoint.pth')
        torch.save(state, filename)
