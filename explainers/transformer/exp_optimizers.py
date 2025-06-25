import torch
import torch.nn as nn

def fuse_conv_bn(conv, bn):
    """
    Fuse Conv2d + BatchNorm2d into a single Conv2d (updates weight and bias)
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True  # We will compute bias
    ).to(conv.weight.device)

    # Prepare fused parameters
    w_conv = conv.weight.clone()
    if conv.bias is not None:
        b_conv = conv.bias.clone()
    else:
        b_conv = torch.zeros(conv.weight.size(0), device=w_conv.device)

    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_eps = bn.eps

    std = torch.sqrt(bn_var + bn_eps)
    w_scale = bn_weight / std
    fused_conv.weight = nn.Parameter(w_conv * w_scale.reshape([-1,1,1,1]))
    fused_conv.bias = nn.Parameter((b_conv - bn_mean)/std * bn_weight + bn_bias)

    return fused_conv


conv = nn.Conv2d(3, 16, 3, padding=1)
bn = nn.BatchNorm2d(16)

# After training: fuse before exporting to TorchScript/TensorRT
fused = fuse_conv_bn(conv, bn)


# model = YourResNetLikeModel().eval()
# scripted = torch.jit.script(model)
# optimized = torch.jit.optimize_for_inference(scripted)
#
#
# import torch_tensorrt
#
# trt_model = torch_tensorrt.compile(
#     model,
#     inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
#     enabled_precisions={torch.float, torch.half, torch.int8},  # mixed precision
# )
#
#
# import torch.quantization
#
# model = YourResNetLikeModel()
# model.eval()
#
# # Fuse modules
# model_fused = torch.quantization.fuse_modules(
#     model,
#     [["conv1", "bn1", "relu"], ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"]]
# )
#
# # Quantize
# model_int8 = torch.quantization.quantize_dynamic(
#     model_fused,
#     {nn.Linear, nn.Conv2d},
#     dtype=torch.qint8
# )
#
#
# out = conv(x)
# out += identity  # skip
# out = relu(out)
#
#
# # Original
# x = F.interpolate(x, scale_factor=2, mode='bilinear')
# x = conv(x)
#
# # Optimized
# x = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)(x)
