import torch
import torch.nn as nn
import torch.nn.functional as functional


class SeparatingStyleAndContent(nn.Module):
    def __init__(self, nsamples: int, C: int = 64):
        super().__init__()
        self.nsamples = nsamples
        self.C = C

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

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def create_relu():
        return nn.LeakyReLU(negative_slope=0.2)

    def create_encoder_layer(self, out_channels, increase_channels=True, first_layer=False, last_layer=False):
        return nn.Sequential(
            nn.Flatten(1, 2) if first_layer else nn.Identity(),
            nn.Conv2d(
                kernel_size=3 if not first_layer else 5,
                stride=2 if not first_layer else 1,
                in_channels=self.nsamples if first_layer else (out_channels // 2 if increase_channels else out_channels),
                out_channels=out_channels,
                dtype=torch.double,
                padding=1
            ),
            SeparatingStyleAndContent.create_relu(),
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
            self.create_relu() if not last_layer else nn.Identity()
        )

    def forward(self, content_ims: torch.Tensor, style_ims: torch.Tensor):
        content_ims1 = self.content1(content_ims)
        content_ims2 = self.content2(content_ims1)
        content_ims3 = self.content3(content_ims2)
        content_ims4 = self.content4(content_ims3)
        content_ims5 = self.content5(content_ims4)
        content_ims6 = self.content6(content_ims5)
        content_ims7 = self.content7(content_ims6)
        content_ims8 = self.content8(content_ims7)

        # print(content_ims1.size())
        # print(content_ims2.size())
        # print(content_ims3.size())
        # print(content_ims4.size())
        # print(content_ims5.size())
        # print(content_ims6.size())
        # print(content_ims7.size())
        # print(content_ims8.size())

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

        # print(result_ims1.size())
        # print(result_ims2.size())
        # print(result_ims3.size())
        # print(result_ims4.size())
        # print(result_ims5.size())
        # print(result_ims6.size())
        # print(result_ims7.size())
        # print(result_ims8.size())

        return self.sigmoid(result_ims8)


def separating_style_and_content_loss(true_img_b: torch.Tensor, output_img_b: torch.Tensor):
    main_loss = nn.L1Loss()
    return main_loss(output_img_b, true_img_b)

