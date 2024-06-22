
#ifndef __OACTS_TBL_H__
#define __OACTS_TBL_H__

/* How to get i'th value
if (oact->log2_scale >= 0)
    val = (float)oact->buf[i] / (1 << oact->log2_scale)
else
    val = (float)oact->buf[i] * (1 << (-oact->log2_scale))
*/

act_tensor_t Conv_206 = {
    .data_type = TYPE_INT8,
    .buf = (void*)0x0, .size = 86400, .fb = 4,
    .num_dimension = 4, .dimensions = {1,18,60,80},
    .log2_scale = 6, .zp = 0
};

act_tensor_t Conv_208 = {
    .data_type = TYPE_INT8,
    .buf = (void*)0x26800, .size = 21600, .fb = 4,
    .num_dimension = 4, .dimensions = {1,18,30,40},
    .log2_scale = 6, .zp = 0
};

act_tensor_t Conv_210_Conv2d = {
    .data_type = TYPE_INT8,
    .buf = (void*)0x30e00, .size = 9600, .fb = 4,
    .num_dimension = 4, .dimensions = {1,18,15,20},
    .log2_scale = 6, .zp = 0
};
/*

oact buffer layout
                            layer       base       size fb
(                      Conv_206,        0x0,    0x15180,  4),
(                      Conv_208,    0x26800,     0x5460,  4),
(               Conv_210_Conv2d,    0x30e00,     0x2580,  4),
*/
#endif /*__OACTS_TBL_H__*/

int yolo_use = 1;
int width = 640;
int height = 480;
