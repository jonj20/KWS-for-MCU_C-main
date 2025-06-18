/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ds_cnn.h"

const q7_t DS_CNN_conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY]=CONV1_WT;
const q7_t DS_CNN_conv1_bias[CONV1_OUT_CH]=CONV1_BIAS;
const q7_t DS_CNN_conv2_ds_wt[CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY]=CONV2_DS_WT;
const q7_t DS_CNN_conv2_ds_bias[CONV1_OUT_CH]=CONV2_DS_BIAS;
const q7_t DS_CNN_conv2_pw_wt[CONV2_OUT_CH*CONV1_OUT_CH]=CONV2_PW_WT;
const q7_t DS_CNN_conv2_pw_bias[CONV2_OUT_CH]=CONV2_PW_BIAS;
const q7_t DS_CNN_conv3_ds_wt[CONV2_OUT_CH*CONV3_DS_KX*CONV3_DS_KY]=CONV3_DS_WT;
const q7_t DS_CNN_conv3_ds_bias[CONV2_OUT_CH]=CONV3_DS_BIAS;
const q7_t DS_CNN_conv3_pw_wt[CONV3_OUT_CH*CONV2_OUT_CH]=CONV3_PW_WT;
const q7_t DS_CNN_conv3_pw_bias[CONV3_OUT_CH]=CONV3_PW_BIAS;
const q7_t DS_CNN_conv4_ds_wt[CONV3_OUT_CH*CONV4_DS_KX*CONV4_DS_KY]=CONV4_DS_WT;
const q7_t DS_CNN_conv4_ds_bias[CONV3_OUT_CH]=CONV4_DS_BIAS;
const q7_t DS_CNN_conv4_pw_wt[CONV4_OUT_CH*CONV3_OUT_CH]=CONV4_PW_WT;
const q7_t DS_CNN_conv4_pw_bias[CONV4_OUT_CH]=CONV4_PW_BIAS;
const q7_t DS_CNN_conv5_ds_wt[CONV4_OUT_CH*CONV5_DS_KX*CONV5_DS_KY]=CONV5_DS_WT;
const q7_t DS_CNN_conv5_ds_bias[CONV4_OUT_CH]=CONV5_DS_BIAS;
const q7_t DS_CNN_conv5_pw_wt[CONV5_OUT_CH*CONV4_OUT_CH]=CONV5_PW_WT;
const q7_t DS_CNN_conv5_pw_bias[CONV5_OUT_CH]=CONV5_PW_BIAS;
const q7_t DS_CNN_final_fc_wt[CONV5_OUT_CH*OUT_DIM]=FINAL_FC_WT;
const q7_t DS_CNN_final_fc_bias[OUT_DIM]=FINAL_FC_BIAS;

DS_CNN* DS_CNN_create()
{
  DS_CNN* nn = malloc(sizeof(DS_CNN));
  nn->scratch_pad = malloc(sizeof(q7_t)*SCRATCH_BUFFER_SIZE);
  nn->buffer1 = nn->scratch_pad;
  nn->buffer2 = nn->buffer1 + (CONV1_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y);
  nn->col_buffer = nn->buffer2 + (CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y);
  nn->frame_len = FRAME_LEN;
  nn->frame_shift = FRAME_SHIFT;
  nn->num_mfcc_features = NUM_MFCC_COEFFS;
  nn->num_frames = NUM_FRAMES;
  nn->num_out_classes = OUT_DIM;
  nn->in_dec_bits = MFCC_DEC_BITS;

  return nn;
}

DS_CNN_free(DS_CNN* nn)
{
  free(nn->scratch_pad);
  
  free(nn);
}


// int DS_CNN_get_frame_len(DS_CNN* nn) {
//   return nn->frame_len;
// }

// int DS_CNN_get_frame_shift(DS_CNN* nn) {
//   return nn->frame_shift;
// }

// int DS_CNN_get_num_mfcc_features(DS_CNN* nn) {
//   return nn->num_mfcc_features;
// }

// int DS_CNN_get_num_frames(DS_CNN* nn) {
//   return nn->num_frames;
// }

// int DS_CNN_get_num_out_classes(DS_CNN* nn) {
//   return nn->num_out_classes;
// }

// int DS_CNN_get_in_dec_bits(DS_CNN* nn) {
//   return nn->in_dec_bits;
// }



void DS_CNN_run_nn(DS_CNN* nn, q7_t* in_data, q7_t* out_data)
{
  //CONV1 : regular convolution
  arm_convolve_HWC_q7_basic_nonsquare(in_data, CONV1_IN_X, CONV1_IN_Y, 1, 
    DS_CNN_conv1_wt, CONV1_OUT_CH, CONV1_KX, CONV1_KY, CONV1_PX, CONV1_PY, CONV1_SX, CONV1_SY, 
    DS_CNN_conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 
    nn->buffer1, CONV1_OUT_X, CONV1_OUT_Y, 
    (q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer1,CONV1_OUT_X*CONV1_OUT_Y*CONV1_OUT_CH);

  //CONV2 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(nn->buffer1,CONV2_IN_X,CONV2_IN_Y,CONV1_OUT_CH,
    DS_CNN_conv2_ds_wt,CONV1_OUT_CH,CONV2_DS_KX,CONV2_DS_KY,CONV2_DS_PX,CONV2_DS_PY,CONV2_DS_SX,CONV2_DS_SY,
    DS_CNN_conv2_ds_bias,CONV2_DS_BIAS_LSHIFT,CONV2_DS_OUT_RSHIFT,
    nn->buffer2,CONV2_OUT_X,CONV2_OUT_Y,(q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer2,CONV2_OUT_X*CONV2_OUT_Y*CONV2_OUT_CH);

  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(nn->buffer2, CONV2_OUT_X, CONV2_OUT_Y, CONV1_OUT_CH, 
    DS_CNN_conv2_pw_wt, CONV2_OUT_CH, 1, 1, 0, 0, 1, 1, 
    DS_CNN_conv2_pw_bias, CONV2_PW_BIAS_LSHIFT, CONV2_PW_OUT_RSHIFT, 
    nn->buffer1, CONV2_OUT_X, CONV2_OUT_Y, (q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer1,CONV2_OUT_X*CONV2_OUT_Y*CONV2_OUT_CH);

  //CONV3 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(nn->buffer1,CONV3_IN_X,CONV3_IN_Y,CONV2_OUT_CH,
    DS_CNN_conv3_ds_wt,CONV2_OUT_CH,CONV3_DS_KX,CONV3_DS_KY,CONV3_DS_PX,CONV3_DS_PY,CONV3_DS_SX,CONV3_DS_SY,
    DS_CNN_conv3_ds_bias,CONV3_DS_BIAS_LSHIFT,CONV3_DS_OUT_RSHIFT,
    nn->buffer2,CONV3_OUT_X,CONV3_OUT_Y,(q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer2,CONV3_OUT_X*CONV3_OUT_Y*CONV3_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(nn->buffer2, CONV3_OUT_X, CONV3_OUT_Y, CONV2_OUT_CH, 
    DS_CNN_conv3_pw_wt, CONV3_OUT_CH, 1, 1, 0, 0, 1, 1, 
    DS_CNN_conv3_pw_bias, CONV3_PW_BIAS_LSHIFT, CONV3_PW_OUT_RSHIFT, 
    nn->buffer1, CONV3_OUT_X, CONV3_OUT_Y, (q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer1,CONV3_OUT_X*CONV3_OUT_Y*CONV3_OUT_CH);

  //CONV4 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(nn->buffer1,CONV4_IN_X,CONV4_IN_Y,CONV3_OUT_CH,
    DS_CNN_conv4_ds_wt,CONV3_OUT_CH,CONV4_DS_KX,CONV4_DS_KY,CONV4_DS_PX,CONV4_DS_PY,CONV4_DS_SX,CONV4_DS_SY,
    DS_CNN_conv4_ds_bias,CONV4_DS_BIAS_LSHIFT,CONV4_DS_OUT_RSHIFT,
    nn->buffer2,CONV4_OUT_X,CONV4_OUT_Y,(q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer2,CONV4_OUT_X*CONV4_OUT_Y*CONV4_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(nn->buffer2, CONV4_OUT_X, CONV4_OUT_Y, CONV3_OUT_CH, 
    DS_CNN_conv4_pw_wt, CONV4_OUT_CH, 1, 1, 0, 0, 1, 1, 
    DS_CNN_conv4_pw_bias, CONV4_PW_BIAS_LSHIFT, CONV4_PW_OUT_RSHIFT, 
    nn->buffer1, CONV4_OUT_X, CONV4_OUT_Y, (q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer1,CONV4_OUT_X*CONV4_OUT_Y*CONV4_OUT_CH);

  //CONV5 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(nn->buffer1,CONV5_IN_X,CONV5_IN_Y,CONV4_OUT_CH,
    DS_CNN_conv5_ds_wt,CONV4_OUT_CH,CONV5_DS_KX,CONV5_DS_KY,CONV5_DS_PX,CONV5_DS_PY,CONV5_DS_SX,CONV5_DS_SY,
    DS_CNN_conv5_ds_bias,CONV5_DS_BIAS_LSHIFT,CONV5_DS_OUT_RSHIFT,
    nn->buffer2,CONV5_OUT_X,CONV5_OUT_Y,(q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer2,CONV5_OUT_X*CONV5_OUT_Y*CONV5_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(nn->buffer2, CONV5_OUT_X, CONV5_OUT_Y, CONV4_OUT_CH, 
    DS_CNN_conv5_pw_wt, CONV5_OUT_CH, 1, 1, 0, 0, 1, 1, 
    DS_CNN_conv5_pw_bias, CONV5_PW_BIAS_LSHIFT, CONV5_PW_OUT_RSHIFT, 
    nn->buffer1, CONV5_OUT_X, CONV5_OUT_Y, (q15_t*)nn->col_buffer, NULL);
  arm_relu_q7(nn->buffer1,CONV5_OUT_X*CONV5_OUT_Y*CONV5_OUT_CH);

  //Average pool
  arm_avepool_q7_HWC_nonsquare (nn->buffer1,CONV5_OUT_X,CONV5_OUT_Y,CONV5_OUT_CH,CONV5_OUT_X,CONV5_OUT_Y,
    0,0,1,1,1,1,NULL,nn->buffer2, AVG_POOL_OUT_LSHIFT);

  arm_fully_connected_q7(nn->buffer2, DS_CNN_final_fc_wt, CONV5_OUT_CH, OUT_DIM, 
    FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT, DS_CNN_final_fc_bias, out_data, (q15_t*)nn->col_buffer);

}


