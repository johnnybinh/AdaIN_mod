import torch.nn as nn
import torch
from torchvision import models
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from function import calc_gram_matrix

decoder = nn.Sequential()

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    # 4_1 -> 3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    # here for 2_1 and onward
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    # 2_1 to 1_1
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class RC(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, pad_size=1, activated=True
    ):
        super().__init__()
        self.R = nn.ReflectionPad2d((in_channel, in_channel, in_channel, in_channel))
        self.C = nn.Conv2d(in_channel, out_channel, kernel_size)
        self.Relu = nn.ReLU()

    def forward(self, x):
        h = self.R(x)
        h = self.C(h)
        if self.activated:
            return self.Relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.RC1 = RC(512, 256, 3, 1)
        self.RC2 = RC(256, 256, 3, 1)
        self.RC3 = RC(256, 256, 3, 1)
        self.RC4 = RC(256, 256, 3, 1)
        self.RC5 = RC(256 * 2, 128, 3, 1)  # 2_1 onward
        self.RC6 = RC(128, 128, 3, 1)
        self.RC7 = RC(128, 64, 3, 1)
        self.RC8 = RC(64, 64, 3, 1)
        self.RC9 = RC(64, 64, 3, 1)

    def forward(self, t1, t2):  # t1 = relu_21, t2 = relu4_1
        h = self.RC1(t2)
        h = self.up(h)
        h = self.RC2(h)
        h = self.RC3(h)
        h = self.RC4(h)  # done relu4_1
        t1 = nn.functional.interpolate(t1, h.shape[2:], mode="nearest")
        h = torch.cat([h, t1], dim=1)  # channel concating
        h = self.RC5(h)
        h = self.up(h)
        h = self.RC6(h)
        h = self.RC7(h)
        h = self.up(h)
        h = self.RC8(h)
        h = self.RC9(h)


# maybe later
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),  # 128 -> 64 channel
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),  # 128 channel
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),  # 256 channel
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),  # 256 -> 512 channel, 3x3 padding
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, device):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        self.decoder = Decoder().to(device)
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, "enc_{:d}".format(i + 1))(input)
        return input

    # extract up to relu2_1 only
    def encode_layer2(self, input):
        for i in range(2):
            input = getattr(self, "enc_{:d}".format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    # def calc_style_loss(self, input, target):
    #     assert input.size() == target.size()
    #     assert target.requires_grad is False
    #     return self.mse_loss(
    #         calc_gram_matrix(input), calc_gram_matrix(target)
    #     )

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(
            input_std, target_std
        )

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        # content_feat = self.encode(content)
        # content_feat_relu2 = self.encode_layer2(content)
        # t = adain(content_feat_relu2, style_feats[1])
        # t = alpha * t + (1 - alpha) * content_feat_relu2
        content_feat_immediate = self.encode_with_intermediate(content)

        # extract at relu2_1
        t1 = adain(content_feat_immediate[1], style_feats[1])
        t1 = alpha * t1 + (1 - alpha) * content_feat_immediate[1]
        # extract at relu4_1
        t2 = adain(content_feat_immediate[-1], style_feats[-1])
        t2 = alpha * t2 + (1 - alpha) * content_feat_immediate[-1]

        g_t = self.decoder(t1, t2)
        g_t_feats = self.encode_with_intermediate(g_t)

        # g_t = self.decoder(t)
        # g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t2)  # compare with lower level
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
