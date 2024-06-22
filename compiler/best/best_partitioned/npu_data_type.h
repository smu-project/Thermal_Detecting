
#ifndef __NPU_DATA_TYPE_H__
#define __NPU_DATA_TYPE_H__
/*
    Openedges NPU data type
*/

#include <stdio.h>

enum {
    TYPE_UINT4, TYPE_INT4,
    TYPE_UINT8, TYPE_INT8,
    TYPE_UINT16, TYPE_INT16,
    TYPE_FP16, TYPE_FP32
};

typedef struct {
    /// data type of weight and bias
    int  data_type;
    /// base address of weight
    void *wbuf;
    /// size of weight
    int  wsize;
    /// addr buffer index of weight
    int  wfb;
    /// base address of bias
    void *bbuf;
    /// size of bias
    int  bsize;
    /// addr buffer index of bias
    int  bfb;
    /// input activation channel
    int  ia_c;
    /// output activation channel
    int  oa_c;
    /// conv kernel height
    int  r;
    /// conv kernel width
    int  s;
    /// conv groups
    int  groups;
} weight_tensor_t;

typedef struct {
    /// data type of tensor
    int  data_type;
    /// base address of tensor
    void *buf;
    /// size of tensor
    int  size;
    /// addr buffer index of tensor
    int  fb;
    /// number of dimension
    int num_dimension;
    /// dimensions of tensor
    int dimensions[4];
    /// scale
    int log2_scale;
    /// zero point
    int zp;
} act_tensor_t;
#endif /*__NPU_DATA_TYPE_H__*/