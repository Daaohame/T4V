import numpy as np
from utils.download_lib import download
import os

def seg_saliency(raw_sal):
    '''seg_saliency
    To segment the raw saliency (type: list, dtype: float)
    Return a list whose dtype is integral
    '''
    sal = []
    for i in raw_sal:
        if i >= 0.1:
            sal.append(1)
        else:
            sal.append(0)
    return sal


##################################    Neural Network Modules    #################################

import torch
import torch.nn as nn
from torch.nn import BatchNorm3d

from gluoncv.torch.model_zoo.action_recognition.i3d_resnet import (
    BasicBlock,
    Bottleneck,
    make_res_layer,
)
from gluoncv.torch.model_zoo.model_store import get_model_file
from gluoncv.torch.engine.config import get_cfg_defaults


class I3D_ResNetV1(nn.Module):
    """ResNet_I3D backbone.
    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.
    Args:
        depth (int): Depth of ResNet, from {18, 34, 50, 101, 152}.
        num_stages (int): ResNet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        num_classes,
        depth,
        num_stages=4,
        pretrained=False,
        pretrained_base=True,
        feat_ext=False,
        num_segment=1,
        num_crop=1,
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        conv1_kernel_t=5,
        conv1_stride_t=2,
        pool1_kernel_t=1,
        pool1_stride_t=2,
        inflate_freq=(1, 1, 1, 1),
        inflate_stride=(1, 1, 1, 1),
        inflate_style="3x1x1",
        nonlocal_stages=(-1,),
        nonlocal_freq=(0, 1, 1, 0),
        nonlocal_cfg=None,
        bn_eval=True,
        bn_frozen=False,
        partial_bn=False,
        frozen_stages=-1,
        dropout_ratio=0.5,
        init_std=0.01,
        norm_layer=BatchNorm3d,
        norm_kwargs=None,
        ctx=None,
        **kwargs,
    ):
        super(I3D_ResNetV1, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError("invalid depth {} for resnet".format(depth))

        self.num_classes = num_classes
        self.depth = depth
        self.num_stages = num_stages
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
        self.feat_ext = feat_ext
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert (
            len(spatial_strides)
            == len(temporal_strides)
            == len(dilations)
            == num_stages
        )
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.inflate_freqs = (
            inflate_freq
            if not isinstance(inflate_freq, int)
            else (inflate_freq,) * num_stages
        )
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = (
            nonlocal_freq
            if not isinstance(nonlocal_freq, int)
            else (nonlocal_freq,) * num_stages
        )
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.frozen_stages = frozen_stages
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        if self.bn_frozen:
            if norm_kwargs is not None:
                norm_kwargs["use_global_stats"] = True
            else:
                norm_kwargs = {}
                norm_kwargs["use_global_stats"] = True

        self.first_stage = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=(conv1_kernel_t, 7, 7),
                stride=(conv1_stride_t, 2, 2),
                padding=((conv1_kernel_t - 1) // 2, 3, 3),
                bias=False,
            ),
            norm_layer(num_features=64, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(pool1_kernel_t, 3, 3),
                stride=(pool1_stride_t, 2, 2),
                padding=(pool1_kernel_t // 2, 1, 1),
            ),
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
        )

        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs["use_global_stats"] = True
            else:
                norm_kwargs = {}
                norm_kwargs["use_global_stats"] = True

        res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            layer_name = "layer{}_".format(i + 1)

            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                norm_layer=norm_layer,
                norm_kwargs=norm_kwargs,
                layer_name=layer_name,
            )
            self.inplanes = planes * self.block.expansion
            res_layers.append(res_layer)

        self.res_layers = nn.Sequential(*res_layers)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

        self.st_avg = nn.AdaptiveAvgPool3d(output_size=1)
        self.dp = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)
        self.head = nn.Sequential(self.dp, self.fc)

        # self-defined for additional output
        self.output_mode = False
        self.up = nn.Upsample(size=(32, 56, 56), mode="trilinear", align_corners=True)
        # self.bn_ft = norm_layer(2179)
        self.inter = nn.Sequential(
            nn.Conv3d(
                in_channels=2179,
                out_channels=1024,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=1024,
                out_channels=512,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 3, 3),
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=256,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=112,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 3, 3),
                bias=True,
            ),
            nn.ReLU(),
        )
        self.avg = nn.AdaptiveAvgPool3d(output_size=(32, 1, 1))
        self.bn_ft2 = nn.BatchNorm1d(112)
        self.fc_ft = nn.Sequential(
            nn.Linear(112, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 2),
        )

        self.inflate_weights()

    def inflate_weights(self):
        """Inflate I3D network with its 2D ImageNet pretrained weights."""

        if not self.pretrained_base:
            raise RuntimeError(
                "I3D models need to be inflated. Please set PRETRAINED_BASE to True in config."
            )

        if self.pretrained_base and not self.pretrained:
            import torchvision

            if self.depth == 50:
                R2D = torchvision.models.resnet50(pretrained=True, progress=True)
            elif self.depth == 101:
                R2D = torchvision.models.resnet101(pretrained=True, progress=True)
            else:
                raise RuntimeError(
                    "We only support ResNet50 and ResNet101 for I3D models at this moment."
                )

            # copy conv1
            conv1 = self.first_stage._modules["0"]
            conv1_bn = self.first_stage._modules["1"]

            conv1.weight.data.copy_(
                torch.unsqueeze(R2D.conv1.weight.data, dim=2).repeat(1, 1, 5, 1, 1)
            )
            conv1_bn.weight.data.copy_(R2D.bn1.weight.data)
            conv1_bn.bias.data.copy_(R2D.bn1.bias.data)
            conv1_bn.running_mean.data.copy_(R2D.bn1.running_mean.data)
            conv1_bn.running_var.data.copy_(R2D.bn1.running_var.data)

            res2 = self.res_layers._modules["0"]
            res3 = self.res_layers._modules["1"]
            res4 = self.res_layers._modules["2"]
            res5 = self.res_layers._modules["3"]

            stages = [res2, res3, res4, res5]

            R2Dlayers = [R2D.layer1, R2D.layer2, R2D.layer3, R2D.layer4]

            for s, _ in enumerate(stages):
                res = stages[s]._modules
                count = 0

                for k, block in res.items():
                    if block.conv1.weight.data.shape[2] > 1:
                        block.conv1.weight.data.copy_(
                            torch.unsqueeze(
                                R2Dlayers[s]._modules[str(k)].conv1.weight.data, dim=2
                            ).repeat(1, 1, 3, 1, 1)
                        )
                    else:
                        block.conv1.weight.data.copy_(
                            torch.unsqueeze(
                                R2Dlayers[s]._modules[str(k)].conv1.weight.data, dim=2
                            )
                        )
                    block.conv2.weight.data.copy_(
                        torch.unsqueeze(
                            R2Dlayers[s]._modules[str(k)].conv2.weight.data, dim=2
                        )
                    )
                    block.conv3.weight.data.copy_(
                        torch.unsqueeze(
                            R2Dlayers[s]._modules[str(k)].conv3.weight.data, dim=2
                        )
                    )

                    block.bn1.weight.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn1.weight.data
                    )
                    block.bn1.bias.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn1.bias.data
                    )
                    block.bn1.running_mean.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn1.running_mean.data
                    )
                    block.bn1.running_var.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn1.running_var.data
                    )

                    block.bn2.weight.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn2.weight.data
                    )
                    block.bn2.bias.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn2.bias.data
                    )
                    block.bn2.running_mean.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn2.running_mean.data
                    )
                    block.bn2.running_var.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn2.running_var.data
                    )

                    block.bn3.weight.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn3.weight.data
                    )
                    block.bn3.bias.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn3.bias.data
                    )
                    block.bn3.running_mean.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn3.running_mean.data
                    )
                    block.bn3.running_var.data.copy_(
                        R2Dlayers[s]._modules[str(k)].bn3.running_var.data
                    )

                    if block.downsample is not None:
                        down_conv = block.downsample._modules["0"]
                        down_bn = block.downsample._modules["1"]

                        down_conv.weight.data.copy_(
                            torch.unsqueeze(
                                R2Dlayers[s]
                                ._modules[str(k)]
                                .downsample._modules["0"]
                                .weight.data,
                                dim=2,
                            )
                        )
                        down_bn.weight.data.copy_(
                            R2Dlayers[s]
                            ._modules[str(k)]
                            .downsample._modules["1"]
                            .weight.data
                        )
                        down_bn.bias.data.copy_(
                            R2Dlayers[s]
                            ._modules[str(k)]
                            .downsample._modules["1"]
                            .bias.data
                        )
                        down_bn.running_mean.data.copy_(
                            R2Dlayers[s]
                            ._modules[str(k)]
                            .downsample._modules["1"]
                            .running_mean.data
                        )
                        down_bn.running_var.data.copy_(
                            R2Dlayers[s]
                            ._modules[str(k)]
                            .downsample._modules["1"]
                            .running_var.data
                        )
                    count += 1
            print("I3D weights inflated from pretrained C2D.")

    def forward(self, x):
        bs, _, _, _, _ = x.shape  # bs, 3, 32, 224, 224
        feature_cache = []
        feature_cache.append(x)
        for no, layer in enumerate(self.first_stage):
            x = layer(x)
            if no == 2:
                feature_cache.append(x)
        feature_cache.append(x)  # bs, 64, 8, 56, 56

        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i == 0:
                x = self.pool2(x)
        # print(x.shape)  # bs, 2048, 4, 7, 7
        feature_cache.append(x)

        if self.output_mode:
            # upsampling
            for i in range(len(feature_cache)):
                tmpx = feature_cache[i]
                feature_cache[i] = self.up(tmpx)
            x = torch.cat(
                (
                    feature_cache[0],
                    feature_cache[1],
                    feature_cache[2],
                    feature_cache[3],
                ),
                dim=1,
            )
            # print(x.shape)  # bs, 2179, 32, 56, 56
            # x = self.bn_ft(x)

            x = self.inter(x)
            # print(x.shape)  # bs, 112, 32, 64, 64

            x = self.avg(x)
            x = x.view(x.shape[0], x.shape[1], x.shape[2])
            # print(x.shape)  # bs, 112, 32

            x = self.bn_ft2(x)
            x = x.transpose(1, 2)
            # print(x.shape)  # bs, 32, 112
            x = self.fc_ft(x)
            # print(x.shape)  # bs, 32, 2

            return x

        else:
            # spatial temporal average
            pooled_feat = self.st_avg(x)
            x = pooled_feat.view(bs, -1)
            if self.feat_ext:
                return x

            x = self.head(x)
            return x


def i3d_resnet50_v1_kinetics400(cfg):
    model = I3D_ResNetV1(
        num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
        depth=50,
        pretrained=cfg.CONFIG.MODEL.PRETRAINED,
        pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
        feat_ext=cfg.CONFIG.INFERENCE.FEAT,
        num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
        num_crop=cfg.CONFIG.DATA.NUM_CROP,
        out_indices=[3],
        inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
        partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
        bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
    )

    if cfg.CONFIG.MODEL.PRETRAINED:
        model.load_state_dict(
            torch.load(
                get_model_file(
                    "i3d_resnet50_v1_kinetics400", tag=cfg.CONFIG.MODEL.PRETRAINED
                )
            ),
            strict=False,
        )  # we added some new layers
    return model


def T4V(cfg, PATH):
    model = I3D_ResNetV1(
        num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
        depth=50,
        pretrained=cfg.CONFIG.MODEL.PRETRAINED,
        pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
        feat_ext=cfg.CONFIG.INFERENCE.FEAT,
        num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
        num_crop=cfg.CONFIG.DATA.NUM_CROP,
        out_indices=[3],
        inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
        partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
        bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
    )
    model.load_state_dict(torch.load(PATH))
    return T4V




################################    Video Analysis Modules    #################################

class myModel:
    def __init__(self, device=None, model="I3D", mode="eval", path=None, config=None):
        assert mode in ["eval", "train"], model in ["I3D", "T4V"]
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if config is None:
            config_file = "./cvconfig/action-recognition/configuration/i3d_resnet50_v1_kinetics400.yaml"
            if not os.path.exists(config_file):
                url = "https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/i3d_resnet50_v1_kinetics400.yaml"
                download(url, os.path.dirname(config_file))          
        else:
            config_file = config
        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_file)
        # model = get_model(cfg).to(device)
        if model == "T4V":
            self.model = T4V(cfg, path)
        else:
            torch.manual_seed(
                0xFFFF
            )  # to fix the random seed to initialize the weights
            self.model = i3d_resnet50_v1_kinetics400(cfg).to(self.device)
            # print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)
        if mode == "eval":
            self.model.output_mode = False
            self.model.eval()
        else:
            self.model.output_mode = True
            self.model.train()
            # net.fc = nn.Linear(net.fc.in_features, 32)
            # nn.init.xavier_uniform_(net.fc.weight)
            for name, param in self.model.named_parameters():
                # if not any(keyword in name for keyword in ["up", "bn_ft", "inter", "fc_ft"]):
                if not any(keyword in name for keyword in ["fc_ft"]):
                    param.requires_grad = False
            for name, param in self.model.named_parameters():
                print(name, param.grad, param.requires_grad)

    def predict(self, clip_input, requires_grad=False):
        """Evaluation API
        Input: transformed video
            dtype: float tensor
            range: [0.0, 1.0)
            shape: [C, n_frames, H, W]
        Output: action prediction vector
        """
        mode_saver = self.model.output_mode
        self.model.output_mode = False

        # if not (torch.is_tensor(clip_input) and clip_input.dtype in [torch.float, torch.double]):
        # clip_input = video_transform(clip_input, self.device, requires_grad)

        if requires_grad:
            pred = self.model(torch.unsqueeze(clip_input, dim=0))[0]
            pred = pred.softmax(dim=0)
        else:
            with torch.no_grad():
                pred = self.model(torch.unsqueeze(clip_input, dim=0))[0]
            pred = pred.softmax(dim=0).cpu()

        self.model.output_mode = mode_saver
        return pred

    def get_saliency(self, clip_input, mode="simple"):
        """Saliency Calculation
        Input: transformed video
            dtype: float tensor
            range: [0.0, 1.0)
            shape: [C, n_frames, H, W]
        """
        assert mode in ["simple", "verbose"], "mode must be simple or verbose"
        mode_saver = self.model.output_mode
        self.model.output_mode = False

        clip_input.requires_grad = True
        x = torch.unsqueeze(clip_input, dim=0)

        pred = self.model(x)[0]
        loss = pred.softmax(0).max()
        loss.backward()
        with torch.no_grad():
            saliency = clip_input.grad.transpose(0, 1)  # switch 1st and 2nd dim
            score = saliency**2
            score = score.sum(dim=3).sum(dim=2).sum(dim=1)
            # score = score.sign() * score.abs().pow(1/3)
            score = score.sqrt()
            score = score.cpu()
            pred = pred.softmax(dim=0).cpu()

        if mode == "verbose":
            ### draw bar plot for saliency of this sample
            x_ref = [0, 2, 4, 12, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
            x = []
            y = []
            # for i in sorted(enumerate(score), key=lambda x:x[1]):
            for i in enumerate(score):
                x.append(i[0] * 2)
                y.append(i[1])
            plt.style.use("ggplot")
            xi = list(range(len(x)))
            color_assign = ["r" if i in x_ref else "b" for i in x]
            plt.bar(xi, y, color=color_assign)
            plt.xticks(xi, x, rotation=80)
            plt.xlabel("No. frame")
            plt.ylabel("Saliency")
            plt.savefig("result/saliency.png", format="png", dpi=300)

            clip = video_transforms.CenterCrop(size=(crop_size, crop_size))(video_data)
            pil_image = torchvision.transforms.ToPILImage()(clip[22])
            visualize_heat(
                pil_image,
                saliency[22].unsqueeze(0).sum(dim=1, keepdim=True),
                "./heatmap.png",
                tile_size=1,
            )

        self.model.output_mode = mode_saver
        return score, pred

        # score = score.transpose(0, 1)
        # for i in sorted(enumerate(score[0].cpu().numpy()), key=lambda x:x[1]):
        #     print(i)
        # print(" channel 1")
        # for i in sorted(enumerate(score[1].cpu().numpy()), key=lambda x:x[1]):
        #     print(i)
        # print(" chanenel 2")
        # for i in sorted(enumerate(score[2].cpu().numpy()), key=lambda x:x[1]):
        #     print(i)
        # saliency = (saliency - saliency.min()) / (
        #     saliency.max() - saliency.min()
        # )
