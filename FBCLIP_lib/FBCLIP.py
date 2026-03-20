

from collections import OrderedDict
from typing import Tuple, Union
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn



class FB_Attention(nn.Module):
    """
    ??????????/?????????? Adaptor
    ?????? CrossAttention ?? BasicTransformerBlock ????
    """
    def __init__(self, inplanes=1024, n_heads=8, d_head=64, bg_weight=0.1, dropout=0.):
        super().__init__()
        self.bg_weight = bg_weight
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = dropout
        
        # ????????????
        inner_dim = d_head * n_heads
        
        # ????????????????????
        self.fg_norm = nn.LayerNorm(inplanes)
        self.fg_to_q = nn.Linear(inplanes, inner_dim, bias=False)
        self.fg_to_k = nn.Linear(inplanes, inner_dim, bias=False)
        self.fg_to_v = nn.Linear(inplanes, inner_dim, bias=False)
        self.fg_to_out = nn.Sequential(nn.Linear(inner_dim, inplanes), nn.Dropout(dropout))
        self.fg_scale = nn.Parameter(torch.tensor(d_head ** -0.5))
        
        # ????????????????????
        self.bg_norm = nn.LayerNorm(inplanes)
        self.bg_to_q = nn.Linear(inplanes, inner_dim, bias=False)
        self.bg_to_k = nn.Linear(inplanes, inner_dim, bias=False)
        self.bg_to_v = nn.Linear(inplanes, inner_dim, bias=False)
        self.bg_to_out = nn.Sequential(nn.Linear(inner_dim, inplanes), nn.Dropout(dropout))
        self.bg_scale = nn.Parameter(torch.tensor(d_head ** -0.5))

        # ????/????????
        self.gate = nn.Sequential(
            nn.Linear(inplanes, inplanes // 2),  # ????????
            nn.ReLU(),
            nn.Linear(inplanes // 2, 1),
            nn.Sigmoid()  # ???? 0~1 ????
        )
        self.norm = nn.LayerNorm(inplanes)

    def _attention(self, x, to_q, to_k, to_v, to_out, scale, norm):

        h = self.n_heads
        x_norm = norm(x)
        
        q = to_q(x_norm)
        k = to_k(x_norm)
        v = to_v(x_norm)

        B, N, _ = q.shape
        q, k, v = [t.view(B, N, h, -1).transpose(1, 2) for t in (q, k, v)]

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = to_out(out)
        
        return out

    

    def forward(self, tokens, text=None):
        """
        tokens: [B, N, D]
        text: ??????????????????????????????
        """
        # ????????????
        gate_weight = self.gate(tokens)  # [B, N, 1], ??????1????????
        fg_tokens = tokens * gate_weight
        bg_tokens = tokens * (1 - gate_weight)

        # ????????????
        fg_out = self._attention(
            fg_tokens, 
            self.fg_to_q, self.fg_to_k, self.fg_to_v, 
            self.fg_to_out, self.fg_scale, self.fg_norm
        )
        
        # ????????????
        bg_out = self._attention(
            bg_tokens, 
            self.bg_to_q, self.bg_to_k, self.bg_to_v, 
            self.bg_to_out, self.bg_scale, self.bg_norm
        )

        # ????????
        out = fg_out + bg_out
        return self.norm(out)





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))
        
    def forward(self, x: torch.Tensor, fearure_layers=None, visual_prompt=None):
        out = []
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0
        for i in range(len(self.resblocks)):
            if i < prefix_len:
                x = torch.cat([visual_prompt[i:i+1].repeat(x.size(0), 1, 1), x], dim=1)
            x = self.resblocks[i](x)
            if i < prefix_len:
                x = x[:, visual_prompt[i:i+1].size(1):]
            if fearure_layers is not None and i+1 in fearure_layers:
                out.append(x)
        if fearure_layers is None:
            return x
        else:
            return out

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

        


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())


    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None):
        x = self.conv1(x)  # shape = [B, C, H/patch, W/patch]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]

        # class token
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)  # [B, L+1, C]

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # ---- ?? ???????????? self.positional_embedding ----
        if side != new_side:
            # ?????? cls ??????
            pos_tokens = self.positional_embedding[1:, :].reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
            # ??????????????
            new_pos = torch.nn.functional.interpolate(pos_tokens, size=(new_side, new_side), mode="bilinear", align_corners=False)
            new_pos = new_pos.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, -1)
            # ???? cls + ???? pos
            pos_emb = torch.cat([self.positional_embedding[:1, :], new_pos], dim=0)
        else:
            pos_emb = self.positional_embedding

        # ??????????
        x = x + pos_emb.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x, feature_layers)
        for i, o in enumerate(out):
            out[i] = o.permute(1, 0, 2)  # LND -> NLD
        return out



# from thop import profile
class FBCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    
    def FB_params(self, args,  device):

       
        self.fb_Attention =  FB_Attention(inplanes=self.visual.proj.shape[0]).to(device)
    
        self.semantic_consistency_weight = getattr(args, 'semantic_consistency_weight', 0.15)
        self.semantic_temperature = getattr(args, 'semantic_temperature', 0.07)

        # --- Token ???????????????? (attention pooling ????????) ---
        hidden_dim = self.token_embedding.weight.shape[-1]
        self.token_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)

        # --- ????????MLP (????attention pooling??????) ---
        proj_dim = 768
        self.text_projection_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768)
        ).to(device)

        # --- ???????????????? ---
        # ??????????????????????????????eot_projected, attn_projected, eot_projected_max??
        # ????softmax??????????????
        self.fusion_weights = nn.Parameter(torch.ones(3)/3 )  # [w1, w2, w3]????????????
        
        # ????????????????????????????????????4??
        # ????????????????????????
        embed_dim = 768
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),  # ????????????
            nn.Sigmoid()
        ).to(device)
        
        # ??????????????????????????
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5

        # ??????????????????
        # ????????????????????????
        # img_tokens??18??????3??????????????????????6??????
        self.scale_weights = nn.Parameter(torch.ones(6) / 6)  # ????6??????
        # layer_selector????????img_tokens??????
        self.layer_selector = nn.Parameter(torch.ones(18))  # ????img_tokens??????  

   

    def encode_text_learn(
        self,
        prompts,                      # [batch, L, D] ?????? state prompt embedding
        tokenized_prompts,            # [batch, L] ?????? token id???????? EOT ??????????
        deep_compound_prompts_text=None,
        normalize: bool = False
    ):
     
        cast_dtype = self.transformer.get_cast_dtype()
        
        # ????????
        if prompts.dim() != 3:
            raise ValueError(f"Expected prompts shape [batch, L, D], got {prompts.shape}")
        if tokenized_prompts is None:
            raise ValueError("tokenized_prompts is required for EOT pooling")
        
        # ????????????
        x = prompts + self.positional_embedding.to(cast_dtype)
        
        # Transformer ????
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # ????????
        x = self.ln_final(x).type(self.dtype)  # [batch, L, D]
        

        eot_indices = tokenized_prompts.argmax(dim=-1)  # [batch] ????EOT token????
        eot_features = x[torch.arange(x.shape[0]), eot_indices]  # [batch, D]
        eot_projected_max = eot_features @ self.text_projection  # [batch, embed_dim]

        pooled = x.mean(dim=1)  # [batch, D]
        eot_projected = pooled @ self.text_projection  # [batch, embed_dim]

        
        # === 2. Attention Pooling (????????????) ===
        # ????attention????????????token??????
        attn_logits = self.token_selector(x)  # [batch, L, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)  # [batch, L, 1]
        attn_pooled = torch.sum(attn_weights * x, dim=1)  # [batch, D]
        attn_projected = self.text_projection_mlp(attn_pooled)  # [batch, embed_dim]
        
        fused_features = (1 * eot_projected + 
                         0.5 * attn_projected + 0.5 * eot_projected_max)

        
        # ??????????
        if normalize:
            fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)
        
        return fused_features



        

    def compute_semantic_consistency_loss(self, text_features, image_features):
        """
        ??????????????????????????????????????????????
        """
        # === Step 1. ???????????????? ===
        if len(image_features.shape) == 3:  # [B, num_tokens, D]
            image_features = image_features.mean(dim=1)
        
        # === Step 2. ????????????CLIP ?????? ===
        text_features = F.normalize(text_features, dim=-1)   # [2, D]
        image_features = F.normalize(image_features, dim=-1) # [B, D]
        
        # === Step 3. ?????? logits ===
        similarity = torch.matmul(image_features, text_features.t()) / self.semantic_temperature  # [B, 2]
        probs = F.softmax(similarity, dim=-1)  # [B, 2]
        
        # === Step 4. ?????? (confidence constraint) ===
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()  # [????]

        # loss =  entropy
        
        # === Step 5. Margin ???? (discriminative constraint) ===
        # ???? normal vs abnormal ??????????
        margin = getattr(self, "semantic_margin", 1)
        diff = (similarity[:, 1] - similarity[:, 0]).abs()
        margin_loss = F.relu(margin - diff).mean()
        
        # === Step 6. ???????? ===
        w_entropy = getattr(self, "w_entropy", 1.0)
        w_margin = getattr(self, "w_margin", 0.5)
        
        loss = w_entropy * entropy + w_margin * margin_loss
        
        return loss * self.semantic_consistency_weight



    def enhanced_text_encoding(self, prompts, tokenized_prompts, compound_prompts_text):
        """
        ??????????????????????????????????
        
        Args:
            image_features: ????????????????????????????
            
        Returns:
            text_features: ??????????????
            consistency_loss: ??????????????
        """
        # ????????????
        # text_features = self.encode_state_prompt()
        text_features = self.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text)
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features

    

    
    def get_trainable_parameters(self):
        params = []
        params += list(self.fb_Attention.parameters()) 
        params += list(self.text_projection_mlp.parameters())       # ????????MLP
        params += list(self.token_selector.parameters())            # token??????
        params.append(self.layer_selector)                          # ???????? (Parameter????)
        params.append(self.scale_weights)                           # ????????
        params.append(self.fusion_weights)
        
   
        return params



    def encode_image(self, image, feature_list = [], ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False):
        # return self.visual(image.type(self.dtype), feature_list, ori_patch = ori_patch, proj_use = proj_use, DPAM_layer = DPAM_layer, ffn = ffn)
        print("????",self.visual(image.type(self.dtype), feature_list)[0].shape)
        # ???? torch.Size([8, 1370, 1024])    
        return self.visual(image.type(self.dtype), feature_list)
#????????????????

    def extract_background_features(self, multi_token_features):
        """
        ??????tokens??????????????????
        Args:
            multi_token_features: [B, num_tokens, feature_dim]
        Returns:
            background_features: [B, feature_dim] ????????????
        """
        B, num_tokens, feature_dim = multi_token_features.shape
        
        # ????1: ???????? - ????????tokens????????
        mean_features = torch.mean(multi_token_features, dim=1)  # [B, feature_dim]
        
        # ????2: ???????? - ??????????????????
        max_features = torch.max(multi_token_features, dim=1)[0]  # [B, feature_dim]
        
        # ????3: ?????????? - ??????????????????????
        # ??????????????????????????????????????????
        alpha = 0.5  # ????????
        background_features = alpha * mean_features + (1 - alpha) * max_features
        
        return background_features
    

    

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        # ????positional_embedding????????x????
        if x.shape[1] != self.positional_embedding.shape[0]:
            # ??????????????????????????????positional_embedding
            if x.shape[1] > self.positional_embedding.shape[0]:
                # ????????
                padding_size = x.shape[1] - self.positional_embedding.shape[0]
                padding = torch.zeros(padding_size, self.positional_embedding.shape[1], 
                                    device=self.positional_embedding.device, dtype=self.positional_embedding.dtype)
                pos_emb = torch.cat([self.positional_embedding, padding], dim=0)
            else:
                # ????????
                pos_emb = self.positional_embedding[:x.shape[1]]
        else:
            pos_emb = self.positional_embedding
            
        x = x + pos_emb.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image, feature_layers=None):
        return self.visual(image.type(self.dtype), feature_layers)

    
    def fg_bg_token_refinement(self, x, view, fg_prob, stride=1):
     

        if view == 'ID':
            return x
        
        if view == 'SEM':
            return self._semantic_feature_enhancement(x, fg_prob)
        elif view == 'SPA':
            return self._spatial_feature_enhancement(x, fg_prob)

    
    def _spatial_feature_enhancement(self, x, fg_prob):
       
        cls_token = x[:, :1, :]
        x_tokens = x[:, 1:, :]
        b, l, c = x_tokens.size()
        h = w = int(math.sqrt(l))

        # ???????????? 2D ????????
        x_2d = x_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b, c, h, w
        patchsize = 5
        stride = 1  
        padding = patchsize // 2
        x_unfold = torch.nn.functional.unfold(x_2d, kernel_size=patchsize,
                                              padding=padding, stride=stride)
        x_unfold = x_unfold.permute(0, 2, 1).reshape(-1, c, patchsize * patchsize).permute(0, 2, 1)  
        # (b*h*w, r*r, c)

        # soft mask ????????
        fg_prob_2d = fg_prob.reshape(b, h, w)
        fg_unfold = torch.nn.functional.unfold(fg_prob_2d.unsqueeze(1), kernel_size=patchsize,
                                               padding=padding, stride=stride)
        fg_unfold = fg_unfold.permute(0, 2, 1).reshape(-1, patchsize * patchsize, 1)  
        bg_unfold = 1 - fg_unfold

        # ???????? token ????????????????????????????????????
        x_bg = x_unfold * bg_unfold
        bg_stability = self._compute_stability_score(x_bg, cls_token)
        bg_weights = torch.softmax(bg_stability, dim=1)
        bg_weighted = x_bg * bg_weights
        bg_count = bg_unfold.sum(dim=1, keepdim=True).clamp(min=1e-8)
        x_bg_agg = torch.sum(bg_weighted, dim=1) / bg_count.squeeze(-1)

        # ???????? token ????????????????????????????????????????????
        x_fg = x_unfold * fg_unfold
        fg_info = self._compute_information_richness(x_fg, cls_token)
        fg_weights = torch.softmax(fg_info, dim=1)
        fg_weighted = x_fg * fg_weights
        fg_count = fg_unfold.sum(dim=1, keepdim=True).clamp(min=1e-8)
        x_fg_agg = torch.sum(fg_weighted, dim=1) / fg_count.squeeze(-1)
        # ????????
        x_agg = x_fg_agg + x_bg_agg
        x_agg = x_agg.reshape(b, -1, c)
        x_out = torch.cat([cls_token, x_agg], dim=1)
        return x_out


    def _semantic_feature_enhancement(self, x, fg_prob):
      
        cls_token = x[:, :1, :]
        x_tokens = x[:, 1:, :]
        b, l, c = x_tokens.size()
     
        fg_weights = fg_prob.unsqueeze(-1) * fg_prob.unsqueeze(1)  # [b, l, l]
        bg_weights = (1 - fg_prob.unsqueeze(-1)) * (1 - fg_prob.unsqueeze(1))  # [b, l, l]
        
     
        fg_cls_sim = torch.cosine_similarity(x_tokens, cls_token, dim=-1)  # [b, l]
        fg_info_richness = 1 - fg_cls_sim  # ??CLS?????? = ????????????
        fg_info_weights = fg_info_richness.unsqueeze(-1) * fg_weights  # [b, l, l]
        fg_info_weights = torch.softmax(fg_info_weights, dim=-1)
     
        bg_stability_weights = fg_cls_sim.unsqueeze(-1) * bg_weights  # [b, l, l]
        bg_stability_weights = torch.softmax(bg_stability_weights, dim=-1)
       
        fg_aggregated = torch.matmul(fg_info_weights, x_tokens)  # [b, l, c]
        
      
        bg_aggregated = torch.matmul(bg_stability_weights, x_tokens)  # [b, l, c]
        

        fg_mask = fg_prob.unsqueeze(-1)  # [b, l, 1]
        bg_mask = 1 - fg_mask
        
  
        fg_part = fg_mask * fg_aggregated
        
    
        bg_part = bg_mask * bg_aggregated
        

        alpha = 0.6
        

        aggregated_tokens = fg_part + bg_part
        final_tokens = alpha * aggregated_tokens + (1 - alpha) * x_tokens
        

        x_out = torch.cat([cls_token, final_tokens], dim=1)
        return x_out

    def _normalize_feat(self,x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def generate_fg_softmask(self, x, prev_tokens=None, alpha=3.0, normal_center=None, smooth=True):
      
        cls_token = x[:, :1, :]
        x_tokens = x[:, 1:, :]
        b, l, c = x_tokens.size()

        local_saliency = self._compute_local_saliency(x_tokens)

        if normal_center is not None:
            # ???? normal_center ?????? (c,) ?? (b, c)
            if normal_center.dim() == 1:
                nc = normal_center.view(1, 1, -1).expand(b, 1, -1)
            elif normal_center.dim() == 2 and normal_center.size(0) == b:
                nc = normal_center.unsqueeze(1)
            else:
                # ????????????????
                nc = x_tokens.mean(dim=1, keepdim=True)
            center_dist = torch.norm(x_tokens - nc, dim=-1)
        else:
            batch_center = x_tokens.mean(dim=1, keepdim=True)
            center_dist = torch.norm(x_tokens - batch_center, dim=-1)

        temporal_change = torch.zeros_like(center_dist)
        if prev_tokens is not None:
            prev_x_tokens = prev_tokens[:, 1:, :]
            if prev_x_tokens.size(1) == l:
                temporal_change = torch.norm(x_tokens - prev_x_tokens, dim=-1)

        cls_sim = torch.nn.functional.cosine_similarity(x_tokens, cls_token, dim=-1)
        cls_inconsistency = 1 - cls_sim

        def safe_minmax(x, dim=1, eps=1e-6):
            # ??????/???????? min-max ??????????????????????
            min_v = x.min(dim=dim, keepdim=True).values
            max_v = x.max(dim=dim, keepdim=True).values
            denom = (max_v - min_v).clamp_min(eps)
            return (x - min_v) / denom

        local_sal_norm = safe_minmax(local_saliency)
        center_dist_norm = safe_minmax(center_dist)
        temp_change_norm = safe_minmax(temporal_change)
        cls_inc_norm = safe_minmax(cls_inconsistency)

        # ?? ???????????????? gating??
        anomaly_score = (
            local_sal_norm * 0.30 +
            center_dist_norm * 0.30 +
            cls_inc_norm * 0.30 +
            temp_change_norm * 0.10
        )

        # ?? ????
        if smooth:
            score_1d = anomaly_score.unsqueeze(1)  # (b,1,l)
            kernel = torch.ones(1, 1, 5, device=anomaly_score.device) / 5.0
            anomaly_score = F.conv1d(F.pad(score_1d, (2, 2), mode='reflect'), kernel).squeeze(1)

        # ?? ???????? {0.5, 1.0}?????????? 1???????? 0.5
        threshold = 0.5  # ?? per-sample ???????????? 0.5 ??????
        fg_prob = torch.where(
            anomaly_score > threshold,
            torch.ones_like(anomaly_score),
            torch.full_like(anomaly_score, 0.5),
        )
        return fg_prob



    def _compute_local_saliency(self, x_tokens):
        """
        ?????????????? - ??????????????tokens??????????
        Args:
            x_tokens: (b, l, c) image tokens
        Returns:
            saliency: (b, l) ??????????????
        """
        b, l, c = x_tokens.size()
        
        # ????????token??????????????
        # ????1D????????????????
        kernel_size = 3
        padding = kernel_size // 2
        
        # ??tokens????????????????????
        x_reshaped = x_tokens.transpose(1, 2)  # (b, c, l)
        
        # ????????????
        local_mean = F.avg_pool1d(x_reshaped, kernel_size=kernel_size, stride=1, padding=padding)
        
        # ????????????????????
        diff = torch.norm(x_reshaped - local_mean, dim=1)  # (b, l)
        
        return diff

    def _compute_temporal_consistency(self, current_tokens, prev_tokens):
        """
        ?????????????? - ??????????????????????
        Args:
            current_tokens: (b, l, c) ????tokens
            prev_tokens: (b, l, c) ??????tokens
        Returns:
            consistency: (b, l) ??????????????
        """
        # ????????????????????
        current_strength = torch.norm(current_tokens, dim=-1)  # (b, l)
        prev_strength = torch.norm(prev_tokens, dim=-1)       # (b, l)
        
        # ????????????????????????????????????
        strength_change = torch.abs(current_strength - prev_strength)
        
        # ??????????????
        current_norm = F.normalize(current_tokens, dim=-1)
        prev_norm = F.normalize(prev_tokens, dim=-1)
        direction_similarity = torch.sum(current_norm * prev_norm, dim=-1)  # (b, l)
        
        # ????????????????????
        consistency = direction_similarity - strength_change  # ???????????????????????? -> ????????
        
        return consistency

    def _compute_stability_score(self, x_unfold, cls_token):
        b_hw, r_r, c = x_unfold.size()
        b = cls_token.size(0)
        cls_expanded = cls_token.unsqueeze(1).expand(b, b_hw//b, 1, c).reshape(b_hw, 1, c)

        x_norm = F.normalize(x_unfold, dim=-1)
        cls_norm = F.normalize(cls_expanded, dim=-1)
        similarity = torch.sum(x_norm * cls_norm, dim=-1, keepdim=True)
        feature_strength = torch.norm(x_unfold, dim=-1, keepdim=True)
        stability = similarity * feature_strength
        return stability

    def _compute_information_richness(self, x_unfold, cls_token):
        b_hw, r_r, c = x_unfold.size()
        b = cls_token.size(0)
        cls_expanded = cls_token.unsqueeze(1).expand(b, b_hw//b, 1, c).reshape(b_hw, 1, c)

        x_norm = F.normalize(x_unfold, dim=-1)
        cls_norm = F.normalize(cls_expanded, dim=-1)
        difference = 1.0 - torch.sum(x_norm * cls_norm, dim=-1, keepdim=True)

        local_mean = torch.mean(x_unfold, dim=1, keepdim=True)
        diversity = torch.norm(x_unfold - local_mean, dim=-1, keepdim=True)
        richness = difference * diversity
        return richness


   

    # ------------------- ???????????????? -------------------
    def FEBG(self, x, view, stride=1):
        fg_prob = self.generate_fg_softmask(x)
        return self.fg_bg_token_refinement(x, view, fg_prob, stride)

    def FEBG_iterative(self, x, view, prev_tokens=None, stride=1):
        fg_prob = self.generate_fg_softmask(x, prev_tokens)
        return self.fg_bg_token_refinement(x, view, fg_prob, stride)

    def FG_BG_Enhancement(self, text_features, img_tokens, complementary_views = [], use_iterative=True):

        FG_BG_Enhancement_features = []
        for layer_idx, img_token in enumerate(img_tokens):
            layer_features = []
            prev_token = img_tokens[layer_idx - 1] if layer_idx > 0 and use_iterative else None
            for view in complementary_views:
                if use_iterative and prev_token is not None:
                    Enhancement_feature = self.FEBG_iterative(img_token, view, prev_token)
                else:
                    Enhancement_feature = self.FEBG(img_token, view)

                layer_features.append(self.fb_Attention(Enhancement_feature, text_features))
            
            FG_BG_Enhancement_features.extend(layer_features)
        return FG_BG_Enhancement_features

    

    
    def Enhance_fg_bg_token(self,text_features, img_tokens):
        """????????????"""
        return self.FG_BG_Enhancement(text_features, img_tokens, complementary_views = ["ID", "SEM", "SPA"])
    
    def encode_all_image(self, image,text_features, args):
        img_tokens = self.encode_image(image, args.feature_layers) 
        cached_tokens = self.Enhance_fg_bg_token(text_features, img_tokens)

       

        processed_img_tokens = []
        reconstruction_errors = []

        if len(cached_tokens) > 1:
            # ???????????????????? token
            num_tokens_to_use = len(cached_tokens)
            multi_token_features = []
            for i in range(num_tokens_to_use):
                b, l, c = cached_tokens[i].shape
                token_subset = cached_tokens[i][:, :l // 2, :]  # ??????????????????????
           
                multi_token_features.append(token_subset)

            multi_token_tensor = torch.cat(multi_token_features, dim=1)  # [B, num_tokens, C]
            background_features = self.extract_background_features(multi_token_tensor)  # [B, C]


        for processed_token in cached_tokens:
            b, l, c = processed_token.shape

            if len(cached_tokens) > 1:
                # ???????????????????? token ??????
                background_feature_expanded = background_features.unsqueeze(1).expand(-1, l, -1)

                # ??????????
                background_similarity = torch.cosine_similarity(
                    processed_token.reshape(-1, c), 
                    background_feature_expanded.reshape(-1, c), 
                    dim=-1
                ).reshape(b, l, 1)

                # ????????
                anomaly_features = processed_token - background_feature_expanded
                anomaly_score = torch.norm(anomaly_features, dim=-1, keepdim=True)  # [B, L, 1]
                reconstruction_errors.append(anomaly_score)

                # ??????????
                similarity_weight = 1 - background_similarity
                enhanced_anomaly_features = anomaly_features * similarity_weight

                # ???????????? & ????????
                # alpha  = 0.7
                alpha  = 0.5
                final_token = alpha * processed_token + (1 - alpha) * enhanced_anomaly_features
            else:
                # ??????????????????????
                final_token = processed_token

            processed_img_tokens.append(final_token @ self.visual.proj )

        return processed_img_tokens, reconstruction_errors

    
    
    
    def FB_encode(self, image, args, prompts, tokenized_prompts, compound_prompts_text):

        text_features = self.enhanced_text_encoding( prompts, tokenized_prompts, compound_prompts_text)


        img_tokens, reconstruction_errors = self.encode_all_image(image,text_features, args)

        image_features_for_consistency = img_tokens[-1]   # [B, num_tokens, embed_dim]
    
        consistency_loss = torch.tensor(0.0, device=text_features.device)
        

        consistency_loss = self.compute_semantic_consistency_loss(
            text_features, image_features_for_consistency
        )

        scores = 0
        anomaly_scores = 0
        scale_idx = 0
        
        for i, (img_token, anomaly_error) in enumerate(zip(img_tokens, reconstruction_errors)):
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = torch.matmul(img_token, text_features.permute(1, 0)) / 0.07
            
        
            scale_idx = i // 3 
            weight = torch.softmax(self.scale_weights, dim=0)[scale_idx]
            scores += weight * score

            anomaly_score = anomaly_error.repeat(1, 1, 2)  # [B, L, 2]
            anomaly_scores += weight * anomaly_score
            
            scale_idx += 1
      
        # ??????????????????????
        combined_scores = scores + 0.5 *  anomaly_scores  # ??????????????????????????
        
        prob = torch.softmax(combined_scores, dim=-1)
        cls_label = prob[:, 0, 1].view(-1)
        predict_map = prob[:, 1:, 1]
        
        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        predict_map = predict_map.reshape(b, 1, h, w)
        return cls_label, predict_map, consistency_loss

       
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text