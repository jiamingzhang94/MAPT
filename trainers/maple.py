import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_m import clip
from clip_m.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip_official

from attack import *

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.image_encoder = torch.nn.Sequential(
        #     transforms.Normalize(
        #         [0.48145466, 0.4578275, 0.40821073],
        #         [0.26862954, 0.26130258, 0.27577711]),
        #     clip_model.visual
        # )
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        if cfg.TRAINER.MAPLE.SURROGATE == "vanilla_model":
            # 设置surrogate model
            self.vanilla_model = VanillaCLIP(
                cfg.MODEL.BACKBONE.NAME,
                classnames,
                cfg.TRAINER.MAPLE.PREC,
                cfg.DATASET.NAME
            ).cuda()

            self.surrogate = self.vanilla_model
        # else:
        #     self.surrogate = self

        # 初始化各种对抗攻击方法
        self._init_attacks(cfg)

    def _init_attacks(self, cfg):
        """初始化攻击方法的默认参数配置"""
        self.attack_configs = {
            'pgd': {
                'eps': getattr(cfg.TRAINER.MAPLE, "EPSILON", 1. / 255),
                'steps': getattr(cfg.TRAINER.MAPLE, "ADV_STEPS", 10),
                'alpha': None  # 可选参数
            },
            'fgsm': {
                'eps': getattr(cfg.TRAINER.MAPLE, "EPSILON", 1. / 255)
            }
            # 可以添加更多攻击方法的默认配置
        }

        # 测试时使用的攻击参数配置
        self.test_attack_configs = {
            'pgd': {
                'eps': getattr(cfg.TRAINER.MAPLE, "TEST_EPSILON", 1. / 255),
                'steps': getattr(cfg.TRAINER.MAPLE, "TEST_STEPS", 100),  # 测试时使用更多步数
                'alpha': cfg.TRAINER.MAPLE.TEST_EPSILON / 4
            },
            'fgsm': {
                'eps': getattr(cfg.TRAINER.MAPLE, "TEST_EPSILON", 1. / 255)
            }
        }

    def _normalize_image(self, image):
        """标准化图像的辅助方法"""
        normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        )
        return normalize(image)

    def forward(self, image, label=None, return_features=False):
        image = self._normalize_image(image)
        features_dict = {}
        
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
        # 获取文本特征
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
        
        # 获取图像特征和中间特征
        if return_features:
            image_features, layer_features = self.image_encoder(
                image.type(self.dtype), 
                shared_ctx,
                deep_compound_prompts_vision,
                return_features=True
            )
            features_dict.update(layer_features)
        else:
            image_features = self.image_encoder(
                image.type(self.dtype), 
                shared_ctx,
                deep_compound_prompts_vision
            )
        
        # 特征归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算logits
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        
        if return_features:
            return logits, features_dict
        return logits

    def generate_adv(self, images, labels, attack_type='pgd', attack_params=None):
        """生成对抗样本"""
        if attack_type not in self.attack_configs:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # 合并默认参数和自定义参数
        params = self.attack_configs[attack_type].copy()
        if attack_params is not None:
            params.update(attack_params)

        # 根据参数创建攻击实例
        if attack_type == 'pgd':
            attack = PGDAttack(
                eps=params['eps'],
                steps=params['steps'],
                alpha=params['alpha']
            )
        elif attack_type == 'fgsm':
            attack = FGSMAttack(
                eps=params['eps']
            )

        if hasattr(self, 'surrogate') and self.surrogate is not None:
            images_adv = attack.generate(
                model=self.surrogate,
                images=images,
                labels=labels,
            )
        else:
            images_adv = attack.generate(
                model=self,
                images=images,
                labels=labels,
            )
        return images_adv

    def forward_adv(self, image, label=None, attack_type='pgd'):
        # 测试时使用测试配置
        image_adv = self.generate_adv(
            image, 
            label, 
            attack_type, 
            attack_params=self.test_attack_configs[attack_type]
        )
        # 使用对抗样本进行前向传播
        return self.forward(image_adv, label)

    def evaluate_robustness(self, image, label, attack_types=None):
        """评估模型在多种攻击下的鲁棒性
        Args:
            image: 输入图像
            label: 标签
            attack_types: 要评估的攻击类型列表，如果为None则评估所有已注册的攻击
        Returns:
            dict: 包含每种攻击方法下的logits
        """
        if attack_types is None:
            attack_types = list(self.attack_configs.keys())

        results = {}
        # 加入清洁样本的结果
        results['clean'] = self.forward(image)

        # 各种攻击方法的结果
        for attack_type in attack_types:
            results[attack_type] = self.forward_adv(image, label, attack_type)

        return results


class VanillaCLIP(nn.Module):
    def __init__(self, backbone_name, classnames, prec="fp32", dataset_name="ImageNet"):
        super().__init__()
        self.model, _ = clip_official.clip.load(backbone_name, device='cpu')
        self.model.cuda()
        self.dtype = torch.float32 if prec == "fp32" else torch.float16
        self.dataset_name = dataset_name
        self.prepare_text_features(classnames)

    def prepare_text_features(self, classnames):
        """根据数据集选择对应模板"""
        CUSTOM_TEMPLATES = {
            "OxfordPets": "a photo of a {}, a type of pet.",
            "OxfordFlowers": "a photo of a {}, a type of flower.",
            "FGVCAircraft": "a photo of a {}, a type of aircraft.",
            "DescribableTextures": "{} texture.",
            "EuroSAT": "a centered satellite photo of {}.",
            "StanfordCars": "a photo of a {}.",
            "Food101": "a photo of {}, a type of food.",
            "SUN397": "a photo of a {}.",
            "Caltech101": "a photo of a {}.",
            "UCF101": "a photo of a person doing {}.",
            "ImageNet": "a photo of a {}.",
            "ImageNetSketch": "a photo of a {}.",
            "ImageNetV2": "a photo of a {}.",
            "ImageNetA": "a photo of a {}.",
            "ImageNetR": "a photo of a {}.",
        }
        template = CUSTOM_TEMPLATES.get(
            self.dataset_name,  # 优先使用数据集对应的模板
            "a photo of a {}"  # 默认后备模板
        )
        print(f"Using template: {template}")

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_prompts)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, label=None):
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        image = normalize(image)
        image_features = self.model.encode_image(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()

        # return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]
        self.adv_train = getattr(cfg.TRAINER.MAPLE, "ADV_TRAIN", True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MAPLE.PREC

        if prec == "amp":
            with autocast():
                loss = self._compute_loss(model, image, label)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = self._compute_loss(model, image, label)

            optim.zero_grad()
            loss.backward()
            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # print(f"Total Gradient Norm (after clipping): {total_norm}")
            optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {"loss": loss.item()}

    def _compute_loss(self, model, image, label):
        # 标准前向传播
        logits, clean_features = model(image, label, return_features=True)
        loss = F.cross_entropy(logits, label)
        
        # 对抗训练
        if self.adv_train:
            adv_images = model.generate_adv(image, label, attack_type='pgd')
            adv_logits, adv_features = model(adv_images, label, return_features=True)
            adv_loss = F.cross_entropy(adv_logits, label)
            
            # 计算AFR特征一致性损失
            consistency_loss = 0
            for key in clean_features:
                if key.startswith('afr_layer_'):
                    clean_feat = clean_features[key]
                    adv_feat = adv_features[key]
                    
                    # 归一化后计算MSE
                    clean_feat = clean_feat / (clean_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    adv_feat = adv_feat / (adv_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    layer_loss = F.mse_loss(clean_feat, adv_feat)
                    consistency_loss += layer_loss
            
            # loss_total = loss + adv_loss * 0.1 + consistency_loss * 1000
            # loss_total = loss * (30 - self.epoch) / 30 + \
            #        adv_loss * (1 - (30 - self.epoch) / 30) * 0.1 + \
            #        consistency_loss * (1 - (30 - self.epoch) / 30) * 1000

            loss_total = loss

            # print(f"Clean loss: {loss.item()}")
            # print(f"Adv loss: {adv_loss.item()}")
            # print(f"Consistency loss: {consistency_loss.item()}")

            return loss_total
        
        return loss

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # self.vanilla_model = self.model.vanilla_model

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # 只更新AFR模块的参数
                if any(key in name for key in [
                    "afr.norm1", "afr.norm2", 
                    "afr.q_proj", "afr.k_proj", "afr.v_proj", 
                    "afr.proj_out", "afr.gate"
                ]):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

                # import torch.nn.init as init
                # if param.dim() > 1:  # 对于张量维度大于1的参数，使用Xavier初始化
                #     init.xavier_uniform_(param)
                # else:  # 对于bias等维度为1的参数，使用常数初始化
                #     init.constant_(param, 0)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    # def forward_backward(self, batch):
    #     image, label = self.parse_batch_train(batch)
    #
    #     model = self.model
    #     optim = self.optim
    #     scaler = self.scaler
    #     prec = self.cfg.TRAINER.MAPLE.PREC
    #
    #     if prec == "amp":
    #         with autocast():
    #             # 标准前向传播
    #             output = model(image, label)
    #             loss = output["loss"]
    #
    #             # 对抗训练
    #             if self.adv_train:
    #                 delta = self.model.PGD(image, label, steps=self.steps)
    #                 adv_output = model(image + delta, label)
    #                 loss = (loss + adv_output["loss"]) / 2
    #
    #         optim.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optim)
    #         scaler.update()
    #
    #     else:
    #         # 标准前向传播
    #         loss = model(image, label)
    #         # loss = output["loss"]
    #
    #         # 对抗训练
    #         if self.adv_train:
    #             delta = self.model.PGD(image, label, steps=self.steps)
    #             adv_loss = model(image + delta, label)
    #             loss = (loss + adv_loss) / 2
    #
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #
    #     if (self.batch_idx + 1) == self.num_batches:
    #         self.update_lr()
    #
    #     loss_summary = {
    #         "loss": loss.item()
    #     }
    #
    #     return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
