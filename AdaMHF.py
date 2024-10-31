import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import initialize_weights, NystromAttention, BilinearFusion, SNN_Block, MultiheadAttention
from torchvision.models import vit_large_patch16_224


class CNNExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(CNNExpert, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, in_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class SNNExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.25):
        super(SNNExpert, self).__init__()
        self.snn1 = SNN_Block(in_dim, hidden_dim, dropout)
        self.snn2 = SNN_Block(hidden_dim, in_dim, dropout)

    def forward(self, x):
        x = self.snn1(x)
        x = self.snn2(x)
        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super(TransLayer, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
        vit_model = vit_large_patch16_224(weights='IMAGENET1K_V1')
        encoder_layer = vit_model.encoder.layers[0]
        self.in_dim = encoder_layer.fc1.in_features
        self.hidden_dim = encoder_layer.fc1.out_features
        self.out_dim = encoder_layer.fc2.out_features

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self._initialize_weights(encoder_layer)

    def _initialize_weights(self, encoder_layer):
        with torch.no_grad():
            self.mlp[0].weight.copy_(encoder_layer.fc1.weight.data)
            self.mlp[0].bias.copy_(encoder_layer.fc1.bias.data)
            self.mlp[2].weight.copy_(encoder_layer.fc2.weight.data)
            self.mlp[2].bias.copy_(encoder_layer.fc2.bias.data)

    def forward(self, x):
        return self.mlp(x)


class PREEG(nn.Module):
    def __init__(self, expert_dims, num_experts):
        super(PREEG, self).__init__()
        self.fixed_mlp = FixedMLP()
        self.experts_g = nn.ModuleList([SNNExpert(expert_dims['input_dim'], expert_dims['hidden_dim']) for _ in range(num_experts)])
        self.gate = nn.Sigmoid()

    def forward(self, x, gated_input):
        out = self.fixed_mlp(x)
        for expert in self.experts_g:
            gated_expert_out = self.gate(gated_input) * expert(out)
            out = out + gated_expert_out
        return out


class PREEP(nn.Module):
    def __init__(self, expert_dims, num_experts):
        super(PREEP, self).__init__()
        self.fixed_mlp = FixedMLP()
        self.experts_p = nn.ModuleList([CNNExpert(expert_dims['input_dim'], expert_dims['hidden_dim']) for _ in range(num_experts)])
        self.gate = nn.Sigmoid()

    def forward(self, x, gated_input):
        out = self.fixed_mlp(x)
        for expert in self.experts_p:
            gated_expert_out = self.gate(gated_input) * expert(out)
            out = out + gated_expert_out
        return out


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k=2, pos='ppeg'):
        super(Transformer_P, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.pree = PREEP(expert_dims={'input_dim': feature_dim, 'hidden_dim': feature_dim}, num_experts=num_experts)
        self.pos_layer = EPEG(dim=feature_dim, epeg_2d=False)

    def forward(self, features):
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.pree(h)
        h = self.layer2(h)
        h = self.pos_layer(h, _H, _W)
        h = self.pree(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k_value=2):
        super(Transformer_G, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.pree = PREEG(expert_dims={'input_dim': feature_dim, 'hidden_dim': feature_dim}, num_experts=num_experts)

    def forward(self, features):
        height = features.shape[1]
        adjusted_height, adjusted_width = int(np.ceil(np.sqrt(height))), int(np.ceil(np.sqrt(height)))
        extra_length = adjusted_height * adjusted_width - height
        padded_tensor = torch.cat([features, features[:, :extra_length, :]], dim=1)
        batch_size = padded_tensor.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).cuda()
        combined_tensor = torch.cat((cls_tokens, padded_tensor), dim=1)
        combined_tensor = self.layer1(combined_tensor)
        combined_tensor = self.pree(combined_tensor)
        combined_tensor = self.layer2(combined_tensor)
        combined_tensor = self.pree(combined_tensor)

        return combined_tensor[:, 0], combined_tensor[:, 1:]


class EPEG(nn.Module):
    def __init__(self, dim, epeg_k=15, epeg_2d=False):
        super(EPEG, self).__init__()
        padding = epeg_k // 2
        self.pe = nn.Conv2d(dim, dim, (epeg_k, 1), padding=(padding, 0), groups=dim) if not epeg_2d else nn.Conv2d(dim, dim, epeg_k, padding=padding, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.pe(cnn_feat).flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


class RouterAlpha(nn.Module):
    def __init__(self, alpha, k):
        super(RouterAlpha, self).__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, tokens):
        top_alpha_k = int(self.alpha * self.k)
        top_values, top_indices = torch.topk(tokens, top_alpha_k, dim=1)
        return top_values, top_indices


class RouterK(nn.Module):
    def __init__(self, alpha, k):
        super(RouterK, self).__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, tokens):
        top_1_minus_alpha_k = int((1 - self.alpha) * self.k)
        _, indices = torch.topk(tokens, self.k, dim=1)
        normally_dropped_tokens = tokens[:, top_1_minus_alpha_k:]
        avg_pooled = torch.mean(normally_dropped_tokens, dim=1, keepdim=True)
        return avg_pooled


class ATSA(nn.Module):
    def __init__(self, in_dim, alpha, k, hidden_dim):
        super(ATSA, self).__init__()
        self.router_alpha = RouterAlpha(alpha, k)
        self.router_k = RouterK(alpha, k)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.concat_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU()
        )

    def forward(self, tokens):
        top_alpha_tokens, _ = self.router_alpha(tokens)
        avg_pooled_tokens = self.router_k(tokens)
        mlp_output_alpha = self.mlp1(top_alpha_tokens)
        softmax_output = self.softmax(mlp_output_alpha)
        concat_output = torch.cat((softmax_output, avg_pooled_tokens), dim=-1)
        concat_processed = self.concat_layer(concat_output)
        final_output = self.mlp2(concat_processed)
        return final_output


class LMF(nn.Module):
    def __init__(self, input_dims, output_dim, rank):
        super(LMF, self).__init__()
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.output_dim = output_dim
        self.rank = rank

        self.audio_factor = nn.Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.video_factor = nn.Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.video_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x):
        batch_size = audio_x.shape[0]
        _audio_h = torch.cat((torch.ones(batch_size, 1).to(audio_x.device), audio_x), dim=1)
        _video_h = torch.cat((torch.ones(batch_size, 1).to(video_x.device), video_x), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_zy = fusion_audio * fusion_video

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class AdaMHF(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat",
                 model_size="small", alpha=0.5, beta=0.5, token_selection_strategy="both",
                 genomic_threshold=0.5, pathomic_threshold=0.5, learning_rate=1e-8, position='ppeg'):
        super(AdaMHF, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.alpha = alpha
        self.beta = beta
        self.token_selection_strategy = token_selection_strategy
        self.genomic_threshold = genomic_threshold
        self.pathomic_threshold = pathomic_threshold
        self.learning_rate = learning_rate
        self.position = position

        self.size_configuration = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }

        # Pathomics Embedding Network
        pathomics_hidden_layers = self.size_configuration["pathomics"][model_size]
        pathomics_layers = []
        for idx in range(len(pathomics_hidden_layers) - 1):
            pathomics_layers.append(nn.Linear(pathomics_hidden_layers[idx], pathomics_hidden_layers[idx + 1]))
            pathomics_layers.append(nn.ReLU())
            pathomics_layers.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*pathomics_layers)

        # Genomic Embedding Network
        genomic_hidden_layers = self.size_configuration["genomics"][model_size]
        genomic_networks = []
        for input_dim in omic_sizes:
            genomic_layers = [SNN_Block(dim1=input_dim, dim2=genomic_hidden_layers[0])]
            for i in range(1, len(genomic_hidden_layers)):
                genomic_layers.append(
                    SNN_Block(dim1=genomic_hidden_layers[i - 1], dim2=genomic_hidden_layers[i], dropout=0.25))
            genomic_networks.append(nn.Sequential(*genomic_layers))
        self.genomics_fc = nn.ModuleList(genomic_networks)

        # Pathomics Transformer
        self.pathomics_encoder = Transformer_P(feature_dim=genomic_hidden_layers[-1], pos=self.position)
        self.pathomics_decoder = Transformer_P(feature_dim=genomic_hidden_layers[-1], pos=self.position)

        # Attention layers
        self.pathomics_to_genomics_attention = MultiheadAttention(embed_dim=256, num_heads=1)
        self.genomics_to_pathomics_attention = MultiheadAttention(embed_dim=256, num_heads=1)

        # Genomics Transformer
        self.genomics_encoder = Transformer_G(feature_dim=genomic_hidden_layers[-1])
        self.genomics_decoder = Transformer_G(feature_dim=genomic_hidden_layers[-1])

        self.token_selector = ATSA()

        self.merging_layers = nn.Sequential(
            nn.Linear(genomic_hidden_layers[-1] * 2, genomic_hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(genomic_hidden_layers[-1], genomic_hidden_layers[-1]),
            nn.ReLU()
        )

        self.classifier = nn.Linear(genomic_hidden_layers[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        # Embedding
        genomic_features = [self.genomics_fc[idx](sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomic_features = torch.stack(genomic_features).unsqueeze(0)

        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # Encoder
        cls_token_pathomics, patch_token_pathomics = self.pathomics_encoder(pathomics_features)
        cls_token_genomics, patch_token_genomics = self.genomics_encoder(genomic_features)

        # Token selection
        if self.token_selection_strategy == "both":
            patch_token_pathomics = self.token_selector(patch_token_pathomics, cls_token_pathomics,
                                                        self.pathomic_threshold)
            patch_token_genomics = self.token_selector(patch_token_genomics, cls_token_genomics, self.genomic_threshold)
        elif self.token_selection_strategy == "P":
            patch_token_pathomics = self.token_selector(patch_token_pathomics, cls_token_pathomics,
                                                        self.pathomic_threshold)
        elif self.token_selection_strategy == "G":
            patch_token_genomics = self.token_selector(patch_token_genomics, cls_token_genomics, self.genomic_threshold)

        # Cross-omics attention
        pathomics_in_genomics, _ = self.pathomics_to_genomics_attention(
            patch_token_pathomics.transpose(1, 0),
            patch_token_genomics.transpose(1, 0),
            patch_token_genomics.transpose(1, 0),
        )
        genomics_in_pathomics, _ = self.genomics_to_pathomics_attention(
            patch_token_genomics.transpose(1, 0),
            patch_token_pathomics.transpose(1, 0),
            patch_token_pathomics.transpose(1, 0),
        )

        # Decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(pathomics_in_genomics.transpose(1, 0))
        cls_token_genomics_decoder, _ = self.genomics_decoder(genomics_in_pathomics.transpose(1, 0))

        combined_pathomics = (cls_token_pathomics + cls_token_pathomics_decoder) / 2
        combined_genomics = (cls_token_genomics + cls_token_genomics_decoder) / 2
        lmf_layer = LMF(input_dims=(256, 256), output_dim=256, rank=4).to(combined_pathomics.device)
        output = lmf_layer(combined_pathomics, combined_genomics)

        output_combined = self.merging_layers(torch.cat(
            (
                combined_pathomics,
                combined_genomics,
            ), dim=1
        ))
        final_output = output_combined + output * self.learning_rate
        logits = self.classifier(final_output)

        hazards = torch.sigmoid(logits)
        survival_function = torch.cumprod(1 - hazards, dim=1)

        return hazards, survival_function, cls_token_pathomics, cls_token_pathomics_decoder, cls_token_genomics, cls_token_genomics_decoder
