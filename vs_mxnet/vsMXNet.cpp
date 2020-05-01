#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

#include <VapourSynth/VapourSynth.h>
#include <VapourSynth/VSHelper.h>

#include "MXDll.h"

#ifdef _MSC_VER
#if defined (_WINDEF_) && defined(min) && defined(max)
#undef min
#undef max
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// no int8 and uint16
inline int VSFormatToMXDtype(const VSFormat *format)
{
    if (format->bitsPerSample == 32 && format->sampleType == stFloat) {
        return 0; // float32
    }

    if (format->bitsPerSample == 64 && format->sampleType == stFloat) {
        return 1; // float64
    }

    if (format->bitsPerSample == 16 && format->sampleType == stFloat) {
        return 2; // float16
    }

    if (format->bitsPerSample == 8 && format->sampleType == stInteger) {
        return 3; // uint8
    }

    if (format->bitsPerSample == 32 && format->sampleType == stInteger) {
        return 4; // int32
    }

    if (format->bitsPerSample == 64 && format->sampleType == stInteger) {
        return 6; // int64
    }

    return -1;
}

struct mxnetData
{
    VSNodeRef *node;
    VSVideoInfo vi;
    int patch_w, patch_h;
    int step_w, step_h;
    int scale;
    int output_w, output_h;
    int outstep_w, outstep_h;
    int frame_w, frame_h;
    int out_elem, in_elem;
    void *srcBuffer, *dstBuffer = nullptr;
    PredictorHandle hPred;
};

std::vector<char> ReadFile(const std::string &file_path)
{
    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        return std::vector<char>();
    }

    ifs.seekg(0, std::ios::end);
    auto length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> buf(length);
    ifs.read(buf.data(), length);
    ifs.close();

    return buf;
}

MXNet mx("libmxnet.dll");

inline int mxForward(mxnetData * VS_RESTRICT d)
{
    int ch = d->vi.format->numPlanes;
    auto imageSize = d->patch_h * d->patch_w * ch;

    if (mx.MXPredSetInput(d->hPred, "data", (float *)d->srcBuffer, imageSize) != 0) {
        return 2;
    }

    if (mx.MXPredForward(d->hPred) != 0) {
        return 2;
    }

    uint32_t output_index = 0;

    uint32_t *shape = nullptr;
    uint32_t shape_len = 0;

    // Get Output Result
    if (mx.MXPredGetOutputShape(d->hPred, output_index, &shape, &shape_len) != 0) {
        return 2;
    }

    uint32_t outputSize = 1;
    for (uint32_t i = 0; i < shape_len; ++i) outputSize *= shape[i];

    if (outputSize != d->output_h*d->output_w*ch) {
        return 1;
    }

    if (mx.MXPredGetOutput(d->hPred, output_index, (float *)d->dstBuffer, outputSize) != 0) {
        return 2;
    }

    return 0;
}

static int process(const VSFrameRef *src, VSFrameRef *dst, mxnetData * VS_RESTRICT d, const VSAPI * vsapi) noexcept
{
    if (d->vi.format->subSamplingH || d->vi.format->subSamplingW) {
        return 3;
    }

    const int ch = d->vi.format->numPlanes;
    const int width  = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);

    std::vector<const uint8_t *> srcp(ch);
    std::vector<int> srcStride(ch);

    std::vector<uint8_t *> dstp(ch);
    std::vector<int> dstStride(ch);

    for (int plane = 0; plane < ch; ++plane) {
        srcp[plane] = vsapi->getReadPtr(src, plane);
        dstp[plane] = vsapi->getWritePtr(dst, plane);

        srcStride[plane] = vsapi->getStride(src, plane);
        dstStride[plane] = vsapi->getStride(dst, plane);
    }

    const int patch_size  = d->patch_w  * d->patch_h  * d->in_elem;
    const int output_size = d->output_w * d->output_h * d->out_elem;
    const int in_stride   = d->patch_w  * d->in_elem;
    const int out_stride  = d->output_w * d->out_elem;

    int x = 0, y = 0;
    while (true) {
        auto sy = std::min(y * d->step_h, height - d->patch_h);
        auto ey = std::min(y * d->step_h + d->patch_h, height);

        while (true) {
            auto sx = std::min(x * d->step_w, width - d->patch_w);
            auto ex = std::min(x * d->step_w + d->patch_w, width);

            for (int plane = 0; plane < ch; ++plane) {
                auto stride = srcStride[plane];
                auto _srcp = srcp[plane] + sx * d->in_elem + sy * stride;
                auto buf = (uint8_t *)d->srcBuffer + patch_size * plane;
                vs_bitblt(buf, in_stride, _srcp, stride, in_stride, d->patch_h);
            }

            if (auto err = mxForward(d)) return err;


            auto dstoff_x = std::min(d->frame_w - d->output_w, x * d->outstep_w);
            auto dstoff_y = std::min(d->frame_h - d->output_h, y * d->outstep_h);

            for (int plane = 0; plane < ch; ++plane) {
                auto stride = dstStride[plane];
                auto _dstp = dstp[plane] + dstoff_x * d->out_elem + dstoff_y * stride;
                auto outbuf = (uint8_t *)d->dstBuffer + output_size * plane;
                vs_bitblt(_dstp, stride, outbuf, out_stride, out_stride, d->output_h);
            }

            if (ex == width) break;
            ++x;
        }

        if (ey == height) break;
        ++y;
        x = 0;
    }

    return 0;
}

static const VSFrameRef *VS_CC mxGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    mxnetData *d = static_cast<mxnetData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);

        const auto error = process(src, dst, d, vsapi);
        if (error != 0) {
            std::string err;

            if (error == 1)
                err = "mxnet: input and target shapes do not match";
            else if (error == 2) {
                err = "mxnet: failed to process: ";
                err += mx.MXGetLastError();
            }
            else if (error == 3)
                err = "mxnet: not support clip format";

            vsapi->setFilterError(err.c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return 0;
}

static void VS_CC mxFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    mxnetData *d = static_cast<mxnetData *>(instanceData);
    vsapi->freeNode(d->node);

    mx.MXPredFree(d->hPred);

    vs_aligned_free(d->srcBuffer);
    vs_aligned_free(d->dstBuffer);

    free(d);
}

static void VS_CC mxInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    mxnetData *d = static_cast<mxnetData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static void VS_CC mxCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    int err;

    mxnetData d{};

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    int ch = d.vi.format->numPlanes;
    int width = d.vi.width, height = d.vi.height;

    try {
        if (!isConstantFormat(&d.vi))
            throw std::string{ "only constant format input supported" };

        if (d.vi.format->subSamplingH || d.vi.format->subSamplingW)
            throw std::string{ "all plane must have the save size" };

        int input_dtype = VSFormatToMXDtype(d.vi.format);
        d.in_elem = d.vi.format->bytesPerSample;

        const char* symbol = vsapi->propGetData(in, "symbol", 0, &err);
        if (err)
            throw std::string{ "\"symbol\" is empty" };

        const char* param = vsapi->propGetData(in, "param", 0, &err);
        if (err)
            throw std::string{ "\"param\" is empty" };

        // Input size
        d.patch_w = int64ToIntS(vsapi->propGetInt(in, "patch_w", 0, &err));
        if (err || d.patch_w > width || d.patch_w == 0)
            d.patch_w = width;

        d.patch_h = int64ToIntS(vsapi->propGetInt(in, "patch_h", 0, &err));
        if (err || d.patch_h > height || d.patch_h == 0)
            d.patch_h = height;

        // Step size
        d.step_w = int64ToIntS(vsapi->propGetInt(in, "step_w", 0, &err));
        if (err || d.step_w == 0)
            d.step_w = d.patch_w;

        d.step_h = int64ToIntS(vsapi->propGetInt(in, "step_h", 0, &err));
        if (err || d.step_h == 0)
            d.step_h = d.patch_h;

        if (d.step_w > width)
            d.step_w = width;

        if (d.step_h > height)
            d.step_h = height;

        // Scale
        d.scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            d.scale = 1;

        // Forward output size
        d.output_w = int64ToIntS(vsapi->propGetInt(in, "output_w", 0, &err));
        if (err || d.output_w == 0)
            d.output_w = d.patch_w * d.scale;

        d.output_h = int64ToIntS(vsapi->propGetInt(in, "output_h", 0, &err));
        if (err || d.output_h == 0)
            d.output_h = d.patch_h * d.scale;

        // Output frame size
        d.frame_w = int64ToIntS(vsapi->propGetInt(in, "frame_w", 0, &err));
        if (err || d.frame_w == 0)
            d.vi.width *= d.scale;
        else
            d.vi.width = d.frame_w;

        d.frame_h = int64ToIntS(vsapi->propGetInt(in, "frame_h", 0, &err));
        if (err || d.frame_h == 0)
            d.vi.height *= d.scale;
        else
            d.vi.height = d.frame_h;

        d.frame_w = d.vi.width;
        d.frame_h = d.vi.height;

        int format = int64ToIntS(vsapi->propGetInt(in, "output_format", 0, &err));
        if (!err)
            d.vi.format = vsapi->getFormatPreset(format, core);

        if (d.vi.format->subSamplingH || d.vi.format->subSamplingW)
            throw std::string{ "all output plane must have the save size" };

        d.out_elem = d.vi.format->bytesPerSample;

        // Output Reconstruct step size
        d.outstep_w = int64ToIntS(vsapi->propGetInt(in, "outstep_w", 0, &err));
        if (err || d.step_w == 0)
            d.outstep_w = d.output_w;

        d.outstep_h = int64ToIntS(vsapi->propGetInt(in, "outstep_h", 0, &err));
        if (err || d.outstep_h == 0)
            d.outstep_h = d.output_h;

        if (d.outstep_w > d.vi.width)
            d.outstep_w = d.vi.width;

        if (d.outstep_h > d.vi.height)
            d.outstep_h = d.vi.height;

        // MXnet Config
        const int ctx = int64ToIntS(vsapi->propGetInt(in, "ctx", 0, &err));

        const int dev_id = int64ToIntS(vsapi->propGetInt(in, "dev_id", 0, &err));

        if (ctx != 1 && ctx != 2 && ctx != 0)
            throw std::string{ "context must be 1(cpu) or 2(gpu)" };

        if (d.patch_w < 1)
            throw std::string{ "patch_w must be greater than or equal to 1" };

        if (d.patch_h < 1)
            throw std::string{ "patch_h must be greater than or equal to 1" };

        if (d.step_w < 1)
            throw std::string{ "step_w must be greater than or equal to 1" };

        if (d.step_h < 1)
            throw std::string{ "step_h must be greater than or equal to 1" };

        if (d.output_w < 1)
            throw std::string{ "output_w must be greater than or equal to 1" };

        if (d.output_h < 1)
            throw std::string{ "output_h must be greater than or equal to 1" };

        if (d.vi.width < 1)
            throw std::string{ "frame_w must be greater than or equal to 1" };

        if (d.vi.height < 1)
            throw std::string{ "frame_h must be greater than or equal to 1" };

        if (d.outstep_w < 1)
            throw std::string{ "outstep_w must be greater than or equal to 1" };

        if (d.outstep_h < 1)
            throw std::string{ "outstep_h must be greater than or equal to 1" };

        if (dev_id < 0)
            throw std::string{ "device id must be greater than or equal to 0" };

        d.srcBuffer = vs_aligned_malloc(d.patch_w * d.patch_h * ch * d.in_elem, 128);
        d.dstBuffer = vs_aligned_malloc(d.output_w * d.output_h * ch * d.out_elem, 128);
        if (!d.srcBuffer || !d.dstBuffer)
            throw std::string{ "malloc failure (buffer)" };

        const std::string pluginPath = vsapi->getPluginPath(vsapi->getPluginById("vs.kice.mxnet", core));
        const std::string dataPath = pluginPath.substr(0, pluginPath.find_last_of('/'));

        auto json_data = ReadFile(symbol);
        if (json_data.empty()) {
            json_data = ReadFile(dataPath + "/mxnet-symbol/" + symbol);
        }

        auto param_data = ReadFile(param);
        if (param_data.empty()) {
            param_data = ReadFile(dataPath + "/mxnet-symbol/" + param);
        }

        if (json_data.empty() || param_data.empty())
            throw std::string{ "Cannot open symbol json file or param data file" };

        d.hPred = nullptr;

        // Parameters
        int dev_type = ctx == 0 ? 1 : 2;
        uint32_t num_input_nodes = 1;

        const char *input_name = vsapi->propGetData(in, "input_name", 0, &err);
        if (err)
            input_name = "data";

        const char* input_key[1] = { input_name };
        const char** input_keys = input_key;

        const uint32_t input_shape_indptr[] = { 0, 4 };
        const uint32_t input_shape_data[4] =
        {
            1,
            static_cast<uint32_t>(ch),
            static_cast<uint32_t>(d.patch_h),
            static_cast<uint32_t>(d.patch_w)
        };

        d.hPred = nullptr;

        if (!mx.IsInit()) {
            mx.LoadDll(nullptr);
        }

        if (!mx.IsInit()) {
            throw std::string{ "Cannot load MXNet. Please check MXNet installation." };
        }

        const char *arg_dtype_names[] = { "data" };
        int arg_dtype[1] = { input_dtype };

        // Create Predictor
        if (mx.MXPredCreateEx(
            json_data.data(), param_data.data(), 
            static_cast<int>(param_data.size()),
            dev_type, dev_id,
            num_input_nodes,
            input_keys, input_shape_indptr, input_shape_data,
            1, arg_dtype_names, arg_dtype,
            &d.hPred) != 0) {
            throw std::string{ "Create MXNet Predictor failed: "} + mx.MXGetLastError();
        }

        if (d.hPred == nullptr) {
            throw std::string{ "Invalid MXNet Predictor:" } + mx.MXGetLastError() + " Please Try to Upgrade MXNet.";
        }
    } catch (const std::string & error) {
        vsapi->setError(out, ("mxnet: " + error).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    mxnetData* data = new mxnetData{ d };
    vsapi->createFilter(in, out, "Predict", mxInit, mxGetFrame, mxFree, fmParallelRequests, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("vs.kice.mxnet", "mx", "Use MXNet to accelerated Image-Processing in VapourSynth", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Predict",
        "clip:clip;"
        "symbol:data;"
        "param:data;"
        "patch_w:int:opt;"
        "patch_h:int:opt;"
        "scale:int:opt;"
        "output_w:int:opt;"
        "output_h:int:opt;"
        "frame_w:int:opt;"
        "frame_h:int:opt;"
        "step_w:int:opt;"
        "step_h:int:opt;"
        "outstep_w:int:opt;"
        "outstep_h:int:opt;"
        "output_format:int:opt;"
        "input_name:data:opt;"
        "ctx:int:opt;"
        "dev_id:int:opt;",
        mxCreate, nullptr, plugin);
}
