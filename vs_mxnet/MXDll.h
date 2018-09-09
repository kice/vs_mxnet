#pragma once

#include "PDll.h"

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief handle to Predictor */
typedef void *PredictorHandle;
/*! \brief handle to NDArray list */
typedef void *NDListHandle;

#include <string>

class MXNet : public PDLL
{
public:
	//declare our functions
	DECLARE_FUNCTION10(int, MXPredCreate, const char*, const void*, int, int, int, mx_uint, const char**, const mx_uint*, const mx_uint*, PredictorHandle*)

	DECLARE_FUNCTION4(int, MXPredSetInput, PredictorHandle, const char*, const mx_float*, mx_uint)
	DECLARE_FUNCTION1(int, MXPredForward, PredictorHandle)
	DECLARE_FUNCTION4(int, MXPredGetOutputShape, PredictorHandle, mx_uint, mx_uint**, mx_uint*)
	DECLARE_FUNCTION4(int, MXPredGetOutput, PredictorHandle, mx_uint, mx_float*, mx_uint)

	DECLARE_FUNCTION1(int, MXPredFree, PredictorHandle)

private:
	//use the class declaration macro
	DECLARE_CLASS(MXNet)
};
