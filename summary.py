import torch
from net import Net
import net

resnet_encoder = net.encoder_resnet
resnet_decoder = net.decoder_resnet
network = Net(resnet_encoder=resnet_encoder, resnet_decoder=resnet_decoder)
network.print_summary()
