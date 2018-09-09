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

```
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

Perforamce
==========

To be tested.

Limitation
==========

1. If patch size is not a divsor of input image, it will overwrite some pixel near the edge and cause some perforamce issues.

2. Padding can be done by other filter. It dose not support patch padding now.

3. It will take long time load MXNet, please wait; or you can open an issue to tell the developer.

4. MXNet needs large commit size, do be careful of your system maxinum commit size. But runtime memory usage is average.

5. MXNet will take some time for cudnn auto tuning for convolution layers every time. set MXNET_CUDNN_AUTOTUNE_DEFAULT=0 to disable it. More info [here](https://mxnet.incubator.apache.org/faq/env_var.html).

Compilation
===========

Only requirement is OpenCV for padding. And there are some code to bypass Vapoursynth plugin loading system, which only works on Windows.

In addition, you can get [`MXNet C predict API`](https://github.com/apache/incubator-mxnet/tree/master/include/mxnet) if you needed. Since the plugins use `LoadLibrary` to load MXNet, you dont have to download this API to compile.
