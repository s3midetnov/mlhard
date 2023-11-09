import torch
import torch.nn as nn
import torch.nn.functional as functional


class SeparatingStyleAndContent(nn.Module):
    def __init__(self, nsamples: int, C: int = 64):
        super().__init__()
        self.nsamples = nsamples
        self.C = C

        self.content_flatten = nn.Flatten(1, 2)
        self.style_flatten = nn.Flatten(1, 2)

        self.style1 = self.create_encoder_layer(C, first_layer=True)
        self.style2 = self.create_encoder_layer(2 * C)
        self.style3 = self.create_encoder_layer(4 * C)
        self.style4 = self.create_encoder_layer(8 * C)
        self.style5 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.style6 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.style7 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.style8 = self.create_encoder_layer(8 * C, increase_channels=False, last_layer=True)

        self.content1 = self.create_encoder_layer(C, first_layer=True)
        self.content2 = self.create_encoder_layer(2 * C)
        self.content3 = self.create_encoder_layer(4 * C)
        self.content4 = self.create_encoder_layer(8 * C)
        self.content5 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.content6 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.content7 = self.create_encoder_layer(8 * C, increase_channels=False)
        self.content8 = self.create_encoder_layer(8 * C, increase_channels=False, last_layer=True)

        self.mixer = nn.Bilinear(8 * C, 8 * C, 8 * C, dtype=torch.double)

        self.decoder1 = self.create_decoder_layer(8 * C, decrease_channels=False, first_layer=True, output_padding=True)
        self.decoder2 = self.create_decoder_layer(8 * C, decrease_channels=False)
        self.decoder3 = self.create_decoder_layer(8 * C, decrease_channels=False)
        self.decoder4 = self.create_decoder_layer(8 * C, decrease_channels=False, output_padding=True)
        self.decoder5 = self.create_decoder_layer(8 * C, output_padding=True)
        self.decoder6 = self.create_decoder_layer(4 * C)
        self.decoder7 = self.create_decoder_layer(2 * C, output_padding=True)
        self.decoder8 = self.create_decoder_layer(C, last_layer=True)

        self.final_conv = nn.Conv2d(
            in_channels=1 + self.nsamples,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=torch.double
        )

        self.sigmoid = nn.Sigmoid()

    def create_encoder_layer(self, out_channels, increase_channels=True, first_layer=False, last_layer=False):
        return nn.Sequential(
            nn.Conv2d(
                kernel_size=3 if not first_layer else 5,
                stride=2 if not first_layer else 1,
                in_channels=self.nsamples if first_layer else (
                    out_channels // 2 if increase_channels else out_channels),
                out_channels=out_channels,
                dtype=torch.double,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels, dtype=torch.double),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten() if last_layer else nn.Identity()
        )

    def create_decoder_layer(
            self, in_channels, decrease_channels=True,
            first_layer=False, last_layer=False, output_padding=False
    ):
        return nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)) if first_layer else nn.Identity(),
            nn.ConvTranspose2d(
                kernel_size=3 if not last_layer else 5,
                stride=2 if not last_layer else 1,
                in_channels=in_channels * 2 if not first_layer else in_channels,
                out_channels=1 if last_layer else (in_channels // 2 if decrease_channels else in_channels),
                dtype=torch.double,
                padding=1,
                output_padding=1 if output_padding else 0
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(in_channels // 2 if decrease_channels else in_channels, dtype=torch.double)
            ) if not last_layer else nn.Identity()
        )

    def forward(self, content_ims: torch.Tensor, style_ims: torch.Tensor):

        content_ims = self.content_flatten(content_ims)
        style_ims = self.style_flatten(style_ims)

        content_ims1 = self.content1(content_ims)
        content_ims2 = self.content2(content_ims1)
        content_ims3 = self.content3(content_ims2)
        content_ims4 = self.content4(content_ims3)
        content_ims5 = self.content5(content_ims4)
        content_ims6 = self.content6(content_ims5)
        content_ims7 = self.content7(content_ims6)
        content_ims8 = self.content8(content_ims7)

        style_ims1 = self.style1(style_ims)
        style_ims2 = self.style2(style_ims1)
        style_ims3 = self.style3(style_ims2)
        style_ims4 = self.style4(style_ims3)
        style_ims5 = self.style5(style_ims4)
        style_ims6 = self.style6(style_ims5)
        style_ims7 = self.style7(style_ims6)
        style_ims8 = self.style8(style_ims7)

        result_ims = self.mixer(content_ims8, style_ims8)
        result_ims1 = self.decoder1(result_ims)
        result_ims2 = self.decoder2(torch.cat(tensors=(result_ims1, content_ims7), dim=1))
        result_ims3 = self.decoder3(torch.cat(tensors=(result_ims2, content_ims6), dim=1))
        result_ims4 = self.decoder4(torch.cat(tensors=(result_ims3, content_ims5), dim=1))
        result_ims5 = self.decoder5(torch.cat(tensors=(result_ims4, content_ims4), dim=1))
        result_ims6 = self.decoder6(torch.cat(tensors=(result_ims5, content_ims3), dim=1))
        result_ims7 = self.decoder7(torch.cat(tensors=(result_ims6, content_ims2), dim=1))
        result_ims8 = self.decoder8(torch.cat(tensors=(result_ims7, content_ims1), dim=1))
        result_final = self.final_conv(torch.cat(tensors=(result_ims8, content_ims), dim=1))

        return self.sigmoid(result_final)


def separating_style_and_content_loss(true_img_b: torch.Tensor, output_img_b: torch.Tensor, epsilon=1.):
    main_loss = torch.sum(nn.L1Loss(reduction='none')(output_img_b, true_img_b), dim=(1, 2, 3))
    mask_black = (true_img_b < 0.99).to(dtype=torch.double)
    n_of_black = torch.sum(mask_black, dim=(1, 2, 3)) + epsilon
    w_st = 1. / n_of_black
    w_b = nn.functional.softmax(torch.sum(true_img_b * mask_black, dim=(1, 2, 3)), dim=0)
    return torch.sum(main_loss * w_st * w_b)


def antiwhite_loss(slope):

    def lossfn(true_img_b, output_img_b):
        diff_b = nn.functional.leaky_relu(input=output_img_b - true_img_b, negative_slope=-slope)
        return torch.sum(diff_b)

    return lossfn


def antiblack_loss(slope):

    def lossfn(true_img_b, output_img_b):
        diff_b = nn.functional.leaky_relu(input=true_img_b - output_img_b, negative_slope=-slope)
        return torch.sum(diff_b)

    return lossfn


def separating_style_and_content_antiwhite_loss(true_img_b: torch.tensor, output_img_b: torch.Tensor):
    diff_b = nn.functional.leaky_relu(input=output_img_b - true_img_b, negative_slope=-0.1)

    main_loss = torch.sum(diff_b, dim=(1, 2, 3))
    mask_black = (true_img_b < 0.99).to(dtype=torch.double)
    w_b = nn.functional.softmax(torch.sum(-true_img_b * mask_black, dim=(1, 2, 3)), dim=0)
    return torch.sum(main_loss * w_b)


def simple_l1_loss(true_img_b: torch.Tensor, output_img_b: torch.Tensor):
    return nn.functional.l1_loss(output_img_b, true_img_b, reduction='mean')
