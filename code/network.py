import torch.nn as nn
import torch

# Here are detailed tips to help you build your model. 
# However, the hint only contains information about the size of the image, 
# and no information about the batch size.
# Therefore, you must consider 'batch' when writing code.
# Batch appears at dim=0 of the tensor.

class ConditioningAugmention(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(ConditioningAugmention, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        ################# Problem 2-(a). #################
        # TODO: Implement [FC + Activation] x1 with nn.Sequential()

        self.layer = nn.Sequential(
          nn.Linear(input_dim, emb_dim*2),
          nn.LeakyReLU(0.2)  # 공식 코드에서는 nn.LeakyReLU(0.2)로 쓰였음
        )

        ################# Problem 2-(a). #################

    def forward(self, x):
        ################# Problem 2-(a). #################
        '''
        Inputs:
            x: CLIP text embedding c_txt
        Outputs:
            condition: augmented text embedding \hat{c}_txt
            mu: mean of x extracted from self.layer. Length : half of output from self.layer 
            log_sigma: log(sigma) of x extracted from self.layer. Length : half of output from self.layer


        TODO:
            calculate: condition = mu + exp(log_sigma) * z, z ~ N(0, 1)
            Use torch.randn() to generate z
        '''
        
        # mu = self.layer(x)
        # log_sigma = self.layer(x)
        output = self.layer(x)
        mu = output[:, :self.emb_dim]
        log_sigma = output[:, self.emb_dim:]
        z = torch.randn_like(mu)
        condition = mu + torch.exp(log_sigma) * z
        ################# Problem 2-(a). #################
        return condition, mu, log_sigma


class ImageExtractor(nn.Module):
    def __init__(self, in_chans):
        super(ImageExtractor, self).__init__()
        self.in_chans = in_chans
        self.out_chans = 3

        ################# Problem 2-(b). #################
        # TODO: Implement [TransposeConv2d + Activation] x1
        # Think about Which activation function is required?

        self.image_net = nn.Sequential(
          # TransposeConv2d의 kernel_size, stride 설정은 어떻게?
          # 이미지 크기는 그대로, 채널 크기는 바꿔주기
          nn.ConvTranspose2d(self.in_chans, self.out_chans, kernel_size=3, stride=1, padding=1, bias=False),
          nn.Tanh()
        )

        ################# Problem 2-(b). #################

    def forward(self, x):

        ################# Problem 2-(b). #################
        '''
        Inputs:
            x: input tensor, shape [C, H, W]
        Outputs:
            out: output image extracted with self.image_net, shape [3, H, W]

        TODO: calculate out
        '''
        
        out = self.image_net(x)

        ################# Problem 2-(b). #################
        return out


class Generator_type_1(nn.Module):
    def __init__(self, in_chans, input_dim):
        super(Generator_type_1, self).__init__()
        self.in_chans = in_chans
        self.input_dim = input_dim

        self.mapping = self._mapping_network()
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 16)

    def _mapping_network(self):
        ################# Problem 2-(c). #################

        # TODO: Implement [FC + BN + LeakyReLU] x1 with nn.Sequential()
        # Change the input tensor dimension [projection_dim + noise_dim] into [Ng * 4 * 4]
        mapping_network = nn.Sequential(
          nn.Linear(self.input_dim, self.in_chans*4*4),
          nn.BatchNorm1d(self.in_chans*4*4),
          nn.LeakyReLU(0.2)
         )

        return mapping_network

        ################# Problem 2-(c). #################

    def _upsample_network(self):
        ################# Problem 2-(c). #################

        # TODO: Implement [ConvTranspose2D + BN + ReLU] x4 with nn.Sequential()
        # Change the input tensor dimension [Ng, 4, 4] into [Ng/16, 64, 64]

        upsample_network = nn.Sequential(
          nn.ConvTranspose2d(self.in_chans, self.in_chans // 2, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans // 2),
          nn.LeakyReLU(0.2),
          nn.ConvTranspose2d(self.in_chans // 2, self.in_chans // 4, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans // 4),
          nn.LeakyReLU(0.2),
          nn.ConvTranspose2d(self.in_chans // 4, self.in_chans // 8, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans // 8),
          nn.LeakyReLU(0.2),
          nn.ConvTranspose2d(self.in_chans // 8, self.in_chans // 16, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans // 16),
          nn.LeakyReLU(0.2),
        )
        return upsample_network

        ################# Problem 2-(c). #################

    def forward(self, condition, noise):
        ################# Problem 2-(c). #################
        '''
        Inputs:
            condition: \hat{c}_txt, shape [projection_dim]
            noise: gaussian noise sampled from N(0, 1), shape [noise_dim]
        Outputs:
            out: tensor extracted from self.upsample_layer, shape [Ng/16, 64, 64]
            out_image: image extracted from self.image_net, shape [3, 64, 64]

        TODO:
            (1) Concat condition and noise (Hint: use torch.cat)
            (2) Use self.mapping and tensor.reshape to change the shape of concated tensor into [Ng, 4, 4]
            (3) Use self.upsample_layer to extract out
            (4) Use self.image_net to extract out_image
        '''

        x = torch.cat([condition, noise], dim=1)
        x = self.mapping(x)
        x = x.reshape(x.size(0), -1, 4, 4)
        out = self.upsample_layer(x)
        out_image = self.image_net(out)
        ################# Problem 2-(c). #################
        return out, out_image


class Generator_type_2(nn.Module):
    def __init__(self, in_chans, condition_dim, num_res_layer, device):
        super(Generator_type_2, self).__init__()
        self.device = device

        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.num_res_layer = num_res_layer

        self.joining_layer = self._joint_conv()
        self.res_layer = nn.ModuleList(
            [self._res_layer() for _ in range(self.num_res_layer)])
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 2)

    def _upsample_network(self):
        ################# Problem 2-(d). #################

        # TODO: Implement [ConvTranspose2D + BN + ReLU] x1 with nn.Sequential()
        # Change the input tensor dimension [C, H, W] into [C/2, 2H, 2W]

        upsample_network = nn.Sequential(
          nn.ConvTranspose2d(self.in_chans, self.in_chans//2, 2, 2, bias=False),
          nn.BatchNorm2d(self.in_chans // 2),
          nn.ReLU()
        )
        return upsample_network

        ################# Problem 2-(d). #################

    def _joint_conv(self):
        ################# Problem 2-(d). #################

        # TODO: Implement [Conv2d + BN + ReLU] x1 with nn.Sequential()
        # Just change the channel size of input tensor into self.in_chans
        # The input channel of joining_layer should consider applying the condition vector as attention.

        joint_conv = nn.Sequential(
          nn.Conv2d(self.in_chans + self.condition_dim, self.in_chans, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans),
          nn.ReLU()
        )
        return joint_conv

        ################# Problem 2-(d). #################

    def _res_layer(self):
        return ResModule(self.in_chans)

    def forward(self, condition, prev_output):
        ################# Problem 2-(d). #################
        '''
        Inputs:
            condition: \hat{c}_txt, shape [projection_dim]
            prev_output: 'out' tensor returned from previous stage generator, shape [C, H, W]
        Outputs:
            out: tensor extracted from self.upsample_layer, shape [C/2, 2H, 2W]
            out_image: image extracted from self.image_net, shape [3, 2H, 2W]

        TODO:
            (1) Reshape condition tensor to have save spatial size (height, width) with prev_output tensor
                using tensor.reshape() and tensor.repeat(). Concat is possible only when the condition tensor is changed to the [H, W] size of prev_output.
            (2) Concat condition tensor from (1) and prev_output along the channel axis
            (3) Use self.upsample_layer to extract out
            (4) Use self.image_net to extract out_image
            Hint: use for loop to inference multiple res_layers
        '''
        res = prev_output.shape[-1]  # spatial size
        reshaped_condition = condition.view(condition.size(0), -1, 1, 1).repeat(1, 1, res, res)
        x = torch.cat([reshaped_condition, prev_output], dim=1)

        # joining_layer
        x = self.joining_layer(x)
        # res_layer
        for res_layers in self.res_layer:
            x = res_layers(x)

        out = self.upsample_layer(x)
        out_image = self.image_net(out)
        ################# Problem 2-(d). #################
        return out, out_image


class ResModule(nn.Module):
    def __init__(self, in_chans):
        super(ResModule, self).__init__()
        self.in_chans = in_chans

        ################# Problem 2-(d). #################

        # TODO: Implement [Conv2d + BN] + ReLU + [Conv2d + BN] with nn.Sequential()

        self.layer = nn.Sequential(
          nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans),
          nn.ReLU(),
          nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(self.in_chans)
        )

        ################# Problem 2-(d). #################

    def forward(self, x):
        ################# Problem 2-(d). #################
        '''
        Inputs:
            x: input tensor, shape [C, H, W]
        Outputs:
            res_out: output tensor, shape [C, H, W]
        TODO: implement residual connection
        '''
        
        res_out = x + self.layer(x)
        ################# Problem 2-(d). #################
        return res_out


class Generator(nn.Module):
    def __init__(self, text_embedding_dim, projection_dim, noise_input_dim, in_chans, out_chans, num_stage, device):
        super(Generator, self).__init__()
        self.device = device

        self.text_embedding_dim = text_embedding_dim
        self.condition_dim = projection_dim
        self.noise_dim = noise_input_dim
        self.input_dim = self.condition_dim + self.noise_dim
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.num_stage = num_stage
        self.num_res_layer_type2 = 2  # NOTE: you can change this

        # return layers
        self.condition_aug = self._conditioning_augmentation()
        self.g_layer = nn.ModuleList(
            [self._stage_generator(i) for i in range(self.num_stage)])

    def _conditioning_augmentation(self):
        # Define conditioning augmentation of conditonal vector introduced in
        # (StackGAN) https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf

        return ConditioningAugmention(self.text_embedding_dim, self.condition_dim, self.device)

    def _stage_generator(self, i):

        ################# Problem 2-(e). #################
        '''
        TODO: return the class instance of Generator_type_1 or Generator_type_2 class
        Hint: Stage i generator's self.in_chans = stage i-1 generator's 'out' tensor's channel size
        '''

        if i == 0:
            return Generator_type_1(self.in_chans, self.input_dim)

        else:
          # 직접 구현했는데 함수를 통해 out 텐서의 채널 크기를 바로 가져오면 더 좋을 듯
            out_channel_size = self.in_chans // 16
            out_channel_size = out_channel_size // (2**(i-1))
            return Generator_type_2(out_channel_size, self.condition_dim, self.num_res_layer_type2, self.device)

        ################# Problem 2-(e). #################

    def forward(self, text_embedding, noise):
        ################# Problem 2-(e). #################
        '''
        Inputs:
            text_embedding: c_txt
            z: gaussian noise sampled from N(0, 1)
        Outputs:
            fake_images: List that containing the all fake images generated from each stage's Generator
            mu: mean of c_txt extracted from CANet
            log_sigma: log(sigma) of c_txt extracted from CANet
        TODO:
            (1) Calculate \hat{c}_txt, mu, log_sigma
            (2) Generate fake_images by inferencing each stage's generator in series (Use for loop)
        '''

        fake_images = []
        condition, mu, log_sigma = self.condition_aug(text_embedding)

        prev_output = None  # To hold the output of the previous stage generator

        # (2) Generate fake_images by inferencing each stage's generator in series (Use for loop)
        for i in range(self.num_stage):
            if i == 0:
                prev_output, fake_img = self.g_layer[i](condition, noise)
            else:
                prev_output, fake_img = self.g_layer[i](condition, prev_output)
            fake_images.append(fake_img)

        # for i in range(self.num_stage):
        #   _, fake_img = self.g_layer[i].forward(condition, noise)
        #   fake_images.append(fake_img)



        return fake_images, mu, log_sigma


class UncondDiscriminator(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UncondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        ################# Problem 3-(b). #################

        # TODO: Implement [Conv2d] with nn.Sequential()
        # Change the input tensor dimension [8Nd, 4, 4] into [1, 1, 1]
        # Use only one Conv2d.

        self.uncond_layer = nn.Sequential(
          nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
          nn.Conv2d(self.in_chans*8, 1, kernel_size=1, stride=1, bias=False)
          # nn.Conv2d(self.in_chans*8, 1, kernel_size=4, stride=1, bias=False)
        )

        ################# Problem 3-(b). #################

    def forward(self, x):
        ################# Problem 3-(b). #################
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
        Outputs:
            uncond_out: output tensor extracted frm self.uncond_layer, shape [1, 1, 1]
        '''
        uncond_out = self.uncond_layer(x)

        ################# Problem 3-(b). #################
        return uncond_out


class CondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, out_chans):
        super(CondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.out_chans = out_chans

        ################# Problem 3-(c). #################

        # TODO: Implement [Conv2d + BN + LeakyReLU] + Conv2d with nn.Sequential()
        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]
        # Hint: [8Nd + projection_dim, 4, 4] -> [8Nd, 4, 4] -> [1, 1, 1]

        self.cond_layer = nn.Sequential(
          nn.Conv2d(self.in_chans*8 + self.condition_dim, self.in_chans*8, kernel_size=1, stride=1, bias=False),
          nn.BatchNorm2d(self.in_chans*8),
          nn.LeakyReLU(0.2),
          nn.Conv2d(self.in_chans*8, 1, kernel_size=4, stride=1, bias=False)
        )

        ################# Problem 3-(c). #################

    def forward(self, x, c):
        ################# Problem 3-(c). #################
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            cond_out: output tensor extracted frm self.cond_layer, shape [1, 1, 1]
        TODO:   
            (1) Change the shape of c into [projection_dim, 4, 4] with tensor.view and tensor.repeat
            (2) Concat x and reshaped c using torch.cat
            (3) Extract cond_out using self.cond_layer
        '''

        c = c.view(c.size(0), -1, 1, 1).repeat(1, 1, 4, 4)
        x_c = torch.cat([x, c], dim=1)
        cond_out = self.cond_layer(x_c)
        ################# Problem 3-(c). #################
        return cond_out


class AlignCondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, text_embedding_dim):
        super(AlignCondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.text_embedding_dim = text_embedding_dim

        ################# Problem 3-(d). #################

        # TODO: Implement [Conv2d + BN + SiLU] + Conv2d with nn.Sequential()
        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]
        # Hint: [8Nd + projection_dim, 4, 4] -> [8Nd, 4, 4] -> [clip_embedding_dim, 1, 1]

        self.align_layer = nn.Sequential(
          nn.Conv2d(self.in_chans*8 + self.condition_dim, self.in_chans*8, kernel_size=1, stride=1, bias=False),
          nn.BatchNorm2d(self.in_chans*8),
          nn.SiLU(),
          nn.Conv2d(self.in_chans*8, self.text_embedding_dim, kernel_size=4, stride=1)
        )

        ################# Problem 3-(d). #################

    def forward(self, x, c):
        ################# Problem 3-(d). #################
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            align_out: output tensor extracted frm self.align_layer, shape [clip_embedding_dim, 1, 1]
        TODO:   
            (1) Change the shape of c into [projection_dim, 4, 4] with tensor.view and tensor.repeat
            (2) Concat x and reshaped c using torch.cat
            (3) Extract align_out using self.align_layer
            (4) Change the shape of [clip_embedding_dim, 1, 1] into [clip_embedding_dim] with tensor.squeeze()            
        '''
        
        c = c.view(c.size(0), -1, 1, 1).repeat(1, 1, 4, 4)
        x_c = torch.cat([x, c], dim=1)
        align_out = self.align_layer(x_c)
        align_out = align_out.squeeze()
        ################# Problem 3-(d). #################
        return align_out


class Discriminator(nn.Module):
    def __init__(self, projection_dim, img_chans, in_chans, out_chans, text_embedding_dim, curr_stage, device):
        super(Discriminator, self).__init__()
        self.condition_dim = projection_dim
        self.img_chans = img_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.text_embedding_dim = text_embedding_dim
        self.curr_stage = curr_stage
        self.device = device
        
        self.h_dim = 64*(2**curr_stage)/16
        self.global_layer = self._global_discriminator()
        self.prior_layer = self._prior_layer()
        self.uncond_discriminator = self._uncond_discriminator()
        self.cond_discriminator = self._cond_discriminator()
        self.align_cond_discriminator = self._align_cond_discriminator()

    def _global_discriminator(self):
        ################# Problem 3-(a). #################

        # TODO: Implement [Conv2d + LeakyReLU] + [Conv2d + BN + LeakyReLU] x 3 with nn.Sequential()
        # Change the input tensor dimension [3, H, W] into [8Nd, H/16, W/16]
        global_discriminator = nn.Sequential(
          nn.Conv2d(self.img_chans, self.in_chans, kernel_size=2, stride=2, bias=False),
          nn.LeakyReLU(0.2),
          nn.Conv2d(self.in_chans, self.in_chans*2, kernel_size=2, stride=2, bias=False),
          nn.BatchNorm2d(self.in_chans*2),
          nn.LeakyReLU(0.2),
          nn.Conv2d(self.in_chans*2, self.in_chans*4, kernel_size=2, stride=2, bias=False),
          nn.BatchNorm2d(self.in_chans*4),
          nn.LeakyReLU(0.2),
          nn.Conv2d(self.in_chans*4, self.in_chans*8, kernel_size=2, stride=2, bias=False),
          nn.BatchNorm2d(self.in_chans*8),
          nn.LeakyReLU(0.2),
        )
        return global_discriminator

        ################# Problem 3-(a). #################

    def _prior_layer(self):
        ################# Problem 3-(a). #################

        # TODO: Change the input tensor dimension [8Nd, H/16, W/16] into [8Nd, 4, 4]
        # For detailed implementation, check guideline pdf.

        if self.curr_stage == 0:
            return nn.Identity()

        # h_dim에 들어갈 값(global_layer의 output 텐서 shape의 두번째를 가져와야 함)
        import math
        k = int(math.log2((self.h_dim) / 4))

        layers = []
        # pdf에는 잘못 설명된 듯
        for _ in range(k):
          layers.append(nn.Conv2d(self.in_chans*8, self.in_chans*8, kernel_size=2, stride=2, bias=False))
          layers.append(nn.BatchNorm2d(self.in_chans*8))
          layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)
    

    def _uncond_discriminator(self):
        return UncondDiscriminator(self.in_chans, self.out_chans)

    def _cond_discriminator(self):
        return CondDiscriminator(self.in_chans, self.condition_dim, self.out_chans)

    def _align_cond_discriminator(self):
        return AlignCondDiscriminator(self.in_chans, self.condition_dim, self.text_embedding_dim)

    def forward(self,
                img,
                condition=None,  # for conditional loss (mu)
                ):

        ################# Problem 3-(a). #################
        '''
        Inputs:
            img: fake/real image, shape [3, H, W]
            condition: mu extracted from CANet, shape [projection_dim]
        Outputs:
            out: fake/real prediction result (common output of discriminator)
            align_out: f_real/f_fake extracted from self.align_cond_discriminator for contrastive learning
        TODO:
            (1) Inference self.global_layer and self.prior_layer in sereis
            (2) If condition is None: only use unconditional discriminator (return align_out = None)
            (3) If condition is not None: use conditional and align discriminator
        Be careful! The final output must be one value!
        '''

        if condition is None:
            out = self.uncond_discriminator(self.prior_layer(self.global_layer(img)))
            align_out = None

        else:
            out = self.cond_discriminator(self.prior_layer(self.global_layer(img)), condition)
            align_out = self.align_cond_discriminator(self.prior_layer(self.global_layer(img)), condition)

        out = nn.Sigmoid()(out)
        ################# Problem 3-(a). #################
        return out, align_out


def weight_init(layer):
    # Do NOT modify
    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, val=0)

    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, val=0.0)
