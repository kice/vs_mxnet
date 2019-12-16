#pragma once

#include "PDll.h"

/*! \brief handle to Predictor */
typedef void *PredictorHandle;

class MXNet : public PDLL
{
public:
    DECLARE_FUNCTION0(const char *, MXGetLastError)

    DECLARE_FUNCTION13(int, MXPredCreateEx,
        const char */*symbol_json_str*/,
            const void */*param_bytes*/,
            int /*param_size*/,
            int /*dev_type*/, int /*dev_id*/,
            const uint32_t /*num_input_nodes*/,
            const char **/*input_keys*/,
            const uint32_t */*input_shape_indptr*/,
            const uint32_t */*input_shape_data*/,
            const uint32_t /*num_provided_arg_dtypes*/,
            const char **/*provided_arg_dtype_names*/,
            const int */*provided_arg_dtypes*/,
            PredictorHandle */*out*/)

    DECLARE_FUNCTION4(int, MXPredSetInput, PredictorHandle, const char*, const float*, uint32_t)
    DECLARE_FUNCTION1(int, MXPredForward, PredictorHandle)
    DECLARE_FUNCTION4(int, MXPredGetOutputShape, PredictorHandle, uint32_t, uint32_t**, uint32_t*)
    DECLARE_FUNCTION4(int, MXPredGetOutput, PredictorHandle, uint32_t, float*, uint32_t)
    DECLARE_FUNCTION1(int, MXPredFree, PredictorHandle)

private:
    //use the class declaration macro
    DECLARE_CLASS(MXNet)
};
