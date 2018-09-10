Description
===========

Use [MXNet](https://github.com/apache/incubator-mxnet) to accelerated Image-Processing in VapourSynth.

Installation
============

You can donwload MSVC Win64 build from [Here](https://github.com/kice/vs_mxnet/releases)

Require MXNet 1.0+

Since MXNet is very large and use many libraries to improve perforamce. We recommend install MXNet via pip.

Install the latest beta build with GPU(CUDA 9.2) support
```
pip install mxnet-cu92 --pre
```
Check here for more infomation [Installing MXNet](http://mxnet.incubator.apache.org/install/index.html?platform=Windows&language=Python&processor=GPU&version=master)

You can check your MXNet installation with.

```
> python -c "import mxnet; print(mxnet.__version__)"
1.3.0
```

You can also check the GPU support of mxnet.

```
> python

>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

**THERE IS NO NEED TO COPY ANY DLLs TO PLUGIN FOLDER OF VAPOURSYNTH EXCEPT THE PLUGIN ITSELF**

Add the follow lines to the beginning of your .vpy file for auto loading dependency

```
import mxnet as mx
import vapoursynth as vs

core = vs.get_core()

if not hasattr(core, 'mx'):
    core.std.LoadPlugin(r'vs_mxnet.dll', altsearchpath=True)

# Your code goes here
```

Python will try to help use load all require dlls (like, MXNet and CUDA). If you delete `core.std.LoadPlugin`, it will still work for vsedit but not work under vspipe.

Usage
=====

    mx.Predict(clip clip, string symbol, string param[, float scale=1.0, int patch_w=0, int patch_h=0, int output_w=128, int output_h=block_w, int frame_w=3, int frame_h=True, int step_w=0, int step_h=0, int outstep_w=0, int outstep_h=0, int padding=0, int border_type=1, int ctx=0, int dev_id=0])

* clip: Clip to process. Only planar format is float sample type of 32 bit depth is supported. RGB and GRAY is supported. YUV is not correctly supported.

* symbol: MXNet symbol json file. If the plugin cannot read the file, it will try to read it from `plugins64\mxnet-symbol\`. You can find more MXNet model [here](https://github.com/WolframRhodium/Super-Resolution-Zoo).

* param: The same as `symbol`, but for model parameters data.

* scale: Set output shape and final frame shape to twice of patch and input clip. It will be ignore if you manully set corresponding parameters.

* patch_w: The horizontal block size for dividing the image during processing. Smaller value results in lower VRAM usage, while larger value may not necessarily give faster speed. The optimal value may vary according to different graphics card and image size. If patch_h is larger than clip's width, it will clamp to clip's width. default: clip's width.

* patch_h: The same as `patch_w` but for vertical. default: clip's height.

* output_w: The horizontal block size for MXNet model output. default: `patch_h` * `scale`.

* output_h: The same as `output_w` but for vertical.

* frame_w: The final output frame size. It dose not have to related to other shapes, like output shape. default: clip's width * `scale`.

* frame_h: The same as `frame_w` but for vertical.

* step_w: The stride of the sliding window for slicing the patch. It will clamp to clip's width if the step larger than it. default: `patch_w`.

* step_h: The same as `step_w` but for vertical.

* outstep_w: The stride of the sliding window for copying the model output to Vapoursynth target frame buffer. It will clamp to output frame' width if the step larger than it. default: `output_w`.

* outstep_h: The same as `outstep_w` but for vertical.

* padding: Add padding to the input clip before feeding the model. It will add a border to all size of the input image. default: 0

* border_type: Same value as [OpenCV BorderTypes](https://docs.opencv.org/3.4.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5). It will be ignored if `padding` is 0. default: `cv::BORDER_REPLICATE`
    * 0: BORDER_CONSTANT              `iiiiii|abcdefgh|iiiiiii`  Only support `i` = 0.
    * 1: BORDER_REPLICATE (default)   `aaaaaa|abcdefgh|hhhhhhh`
    * 2: BORDER_REFLECT               `fedcba|abcdefgh|hgfedcb`
    * 3: BORDER_WRAP                  `cdefgh|abcdefgh|abcdefg`
    * 4: BORDER_REFLECT_101           `gfedcb|abcdefgh|gfedcba`
    * 5: BORDER_TRANSPARENT           `uvwxyz|abcdefgh|ijklmno`

* ctx: Specifies which type of device to use. If GPU was chosen, cuDNN will be used by defalut.
    * 1 = CPU
    * 2 = GPU

* dev_id: Which device to use. Starting with 0.

Example
=======

```
# Place Symbol file and params data into `plugins64\mxnet-symbol\` or use the full path of the files.

symbol = 'Some2x-symbol.json'
param  = 'Some2x-0000.params'
patch_w, patch_h = 400, 300
pad = 7

# Set input size
clip = core.resize.Bicubic(src, 960, 540)

# run some 2x upsampling model with patch size 400x300. Output size will be 1920x1080
sr2x = core.mx.Predict(src, symbol='Some2x-symbol.json', param='Some2x-0000.params', patch_w=patch_w, patch_h=patch_h, scale=2, ctx=2, dev_id=1)

# run Waifu2x 2x upconv model with patch size=400x300 on second GPU, output size is 1920x1080
waifu2x = core.mx.Predict(clip, symbol=r'noise0_scale2.0x_model-symbol.json', 
                     param=r'noise0_scale2.0x_model-0000.params', 
                     patch_w=patch_w+pad*2, patch_h=patch_h+pad*2, 
                     output_w=patch_w*2,    output_h=patch_h*2, 
                     frame_w=1920,          frame_h=1080, 
                     step_w=patch_w,        step_h=patch_h, 
                     padding=pad, ctx=2, dev_id=1, scale=2)

# For multi-GPU processing (scales almost linearly). Only support data parallel now.

even = core.mx.Predict(core.std.SelectEvery(clip, 2, 0), symbol=symbol, param=param, patch_w=patch_w, patch_h=patch_h, scale=2, ctx=2, dev_id=0)
odd = core.mx.Predict(core.std.SelectEvery(clip, 2, 1), symbol=symbol, param=param, patch_w=patch_w, patch_h=patch_h, scale=2, ctx=2, dev_id=1)

res = core.std.Interleave([even, odd])
```

Also see [muvsfunc](https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/super_resolution_mxnet.vpy)'s example.

Perforamce
==========

Here is the conclusion, generally MXNet is faster than Caffe with cuDNN enabled if the bottleneck is not GPU.

If you found that your GPU is not under full load while using Caffe, you can get significant perforamce boost by switching to MXNet. Or your GPU memory is small, you can also switch to MXNet for higher efficiency.

In this test, a 1280x720 RGB image was used as input image and resized by `resize.Bicubic` if needed.

| Model                  | Input Size | Patch Size | Output Size | Speed(fps) | VRAM Usage(MB) | Backend                           |
|------------------------|------------|------------|-------------|------------|----------------|-----------------------------------|
| waifu2x UpRGB          | 1280x720   | 256x256    | 2560x1440   | 7.03       | 543            | MXNet 1.3.0                       |
| waifu2x UpRGB          | 1280x720   | 1280x720   | 2560x1440   | 7.85       | 1815           | MXNet 1.3.0                       |
| waifu2x UpRGB          | 1280x720   | 640x360    | 2560x1440   | 7.03       | 788            | MXNet 1.3.0                       |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 21.74      | 958            | MXNet 1.3.0                       |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 24.54      | 1476           | MXNet 1.3.0 (2 Queues)            |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 41.66      | 958 *2         | MXNet 1.3.0 (2 GPUs)              |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 47.7       | 1476 *2        | MXNet 1.3.0 (4 Queues 2 GPUs)     |
| waifu2x UpRGB          | 960x540    | 960x540    | 1920x1080   | 14.8       | 1216           | MXNet 1.3.0                       |
| waifu2x UpRGB          | 1920x1080  | 1920x1080  | 3840x2160   | 3.60       | 3527           | MXNet 1.3.0                       |
| waifu2x UpRGB          | 1280x720   | 256x256    | 2560x1440   | 2.93       | 527            | Caffe w/ cuDNN                    |
| waifu2x UpRGB          | 1280x720   | 1280x720   | 2560x1440   | 3.11       | 2726           | Caffe w/ cuDNN                    |
| waifu2x UpRGB          | 1280x720   | 640x360    | 2560x1440   | 3.08       | 959            | Caffe w/ cuDNN                    |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 8.48       | 1622           | Caffe w/ cuDNN                    |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 19.6       | 5976           | Caffe w/ cuDNN (6 Queues)         |
| waifu2x UpRGB          | 720x480    | 720x480    | 1440x960    | 32.8       | 5949 *2        | Caffe w/ cuDNN (12 Queues 2 GPUs) |
| waifu2x UpRGB          | 960x540    | 960x540    | 1920x1080   | 5.31       | 1699           | Caffe w/ cuDNN                    |
| waifu2x UpRGB          | 1920x1080  | 960x540    | 3840x2160   | 1.35       | 2254           | Caffe w/ cuDNN                    |
| waifu2x RGB            | 1280x720   | 1280x720   | 2560x1440   | 1.01       | 1752           | OpenCL (CUDA)                     |
| waifu2x RGB            | 1280x720   | 1280x720   | 2560x1440   | 0.93       | 1749           | OpenCL (OpenCL)                   |
| waifu2x RGB            | 1280x720   | 1280x720   | 2560x1440   | 0.93       | N/A            | OpenCL (CPU)                      |
| waifu2x RGB            | 1280x720   | 1280x720   | 2560x1440   | 1.82       | 1999           | Caffe w/ cuDNN                    |
| waifu2x RGB            | 1280x720   | 1280x720   | 2560x1440   | 3.36       | 1442           | MXNet 1.3.0                       |
| waifu2x RGB            | 2560x1440* | 2560x1440  | 2560x1440   | 3.22       | 5155           | MXNet 1.3.0                       |
| EDSR 2x                | 1280x720   | 1280x720   | 2560x1440   | 2.59       | 2732           | MXNet 1.3.0                       |
| EDSR 2x                | 960x540    | 960x540    | 1920x1080   | 4.59       | 1732           | MXNet 1.3.0                       |
| RCAN 2x                | 1280x720   | 1280x720   | 2560x1440   | 0.185      | 3015           | MXNet 1.3.0                       |
| RCAN 2x                | 960x540    | 960x540    | 1920x1080   | 0.324      | 1916           | MXNet 1.3.0                       |
| VDSR 2x (Y only)       | 2560x1440* | 2560x1440  | 2560x1440   | 1.64       | 7697           | MXNet 1.3.0                       |
| VDSR 2x (Y only)       | 1920x1080* | 1920x1080  | 1920x1080   | 2.96       | 5857           | MXNet 1.3.0                       |
| LapSRN 2x (Y only)     | 1280x720   | 1280x720   | 2560x1440   | 5.67       | 3310           | MXNet 1.3.0                       |
| LapSRN 2x (Y only)     | 960x540    | 960x540    | 1920x1080   | 10.47      | 1474           | MXNet 1.3.0                       |
| LapSRN 4x (Y only)     | 960x540    | 960x540    | 3840x2160   | 2.15       | 4565           | MXNet 1.3.0                       |
| DRRN_B1U9 2x  (Y only) | 2560x1440* | 2560x1440  | 2560x1440   | 0.496      | 5898           | MXNet 1.3.0                       |
| DRRN_B1U9 2x  (Y only) | 1920x1080* | 1920x1080  | 1920x1080   | 0.89       | 3514           | MXNet 1.3.0                       |
| DRRN_B1U25 2x (Y only) | 1920x1080* | 1920x1080  | 1920x1080   | 0.316      | 4300           | MXNet 1.3.0                       |
| DBPN 2x                | 640x360    | 640x360    | 1280x720    | 1.21       | 4987           | MXNet 1.3.0                       |
| DBPN 2x                | 960x540    | 480x540    | 1920x1080   | 0.523      | 8090           | MXNet 1.3.0                       |

* All cuDNN version is 7.

* MXNet is using CUDA 9.2. (Version: mxnet_cu92-1.3.0b20180908)

* For some models have the same the shape of output as the input, like Waifu2x RGB, we first resize/upscale the input image to target size by Bicubic, then feed into the model.

* During testing, Waifu2x-Caffe is only utilizing around 30% of the GPU. By increasing the queues depth, we can have a significant boost; but it will take more resources and still slower than MXNet.

* [Waifu2x-Caffe](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Waifu2x-caffe) is using CUDA 9.0.

* OpenCL of Waifu2x implementation is [VapourSynth-Waifu2x-w2xc](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Waifu2x-w2xc).

* All MXNet model in this test can be accessed [here](https://github.com/WolframRhodium/Super-Resolution-Zoo).

Here is the test code:

```
import mxnet
import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf

core = vs.get_core(threads=20)

if not hasattr(core, 'mx'):
    core.std.LoadPlugin(r'vs_mxnet.dll', altsearchpath=True)

# How many frame to run
frames = 600

symbol = r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model-symbol.json'
param = r'waifu2x\upconv_7_anime_style_art_rgb\scale2.0x_model-0000.params'

src = core.lsmas.LWLibavSource(r'test.png', threads=1)
src = core.std.AssumeFPS(src, fpsnum=24000, fpsden=1001)

# If the model is only support Y channel, enable the following lines
#src = mvf.ToYUV(src, css='444', depth=32)
#src = core.std.ShufflePlanes(src, 0, vs.GRAY)
#src = core.resize.Bicubic(src, 720, 480)

src = core.resize.Bicubic(src, 720, 480, format=vs.RGBS)
src = core.std.Loop(src, frames)

block_w = src.width
block_h = src.height

scale = 2

# Waifu2x need to set pad=7, other model dose not have to set padding
pad = 0

def process(clip, gpu):
    return core.mx.Predict(clip, symbol=symbol, param=param,
                         patch_w  = block_w + pad*2,  patch_h  = block_h + pad*2,
                         output_w = block_w*scale,    output_h = block_h*scale,
                         frame_w  = clip.width*scale, frame_h  = clip.height*scale,
                         step_w   = block_w,          step_h   = block_h,
                         padding = pad, ctx = 2, dev_id = gpu)

queue_size = 3
gpus = 2

res = []
for i in range(queue_size):
    part = process(core.std.SelectEvery(src, queue_size, i), i % gpus)
    res.append(part)

flt = core.std.Interleave(res)
flt.set_output()
```

Limitation
==========

1. If patch size is not a divsor of input image, it will overwrite some pixel near the edge and cause some perforamce issues.

2. Padding can be done by other filter. It dose not support patch padding now.

3. It will take long time load MXNet, please wait; or you can open an issue to tell the developer.

4. MXNet needs large commit size, do be careful of your system maxinum commit size. But runtime memory usage is average.

5. MXNet will take some time for cudnn auto tuning for convolution layers every time. set MXNET_CUDNN_AUTOTUNE_DEFAULT=0 to disable it. More info [here](https://mxnet.incubator.apache.org/faq/env_var.html).

6. Please remember that during feeding the first frame, MXNet will allocate very large VRAM block, you might get **Out of Memory** error. Please reduce the patch size to solve it.

7. You might need to restart the program (e.g. vsedit) after you changing the input model file.

Compilation
===========

Only requirement is OpenCV for padding. And there are some code to bypass Vapoursynth plugin loading system, which only works on Windows.

In addition, you can get [`MXNet C predict API`](https://github.com/apache/incubator-mxnet/tree/master/include/mxnet) if you needed. Since the plugins use `LoadLibrary` to load MXNet, you dont have to download this API to compile.
