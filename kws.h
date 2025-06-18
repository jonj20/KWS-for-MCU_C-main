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

#ifndef __KWS_DS_CNN_H__
#define __KWS_DS_CNN_H__


#include "arm_math.h"
#include "mfcc.h"
#include "ds_cnn.h"

typedef struct {
//public:
  int16_t* audio_buffer;
  q7_t *mfcc_buffer;
  q7_t *output;
  q7_t *predictions;
  q7_t *averaged_output;
  int num_frames;
  int num_mfcc_features;
  int frame_len;
  int frame_shift;
  int num_out_classes;
  int audio_block_size;
  int audio_buffer_size;
//protected:
  MFCC *mfcc;
  DS_CNN *nn;
  int mfcc_buffer_size;
  int recording_win;
  int sliding_window_len;
}KWS;


KWS* KWS_create_1(int recording_win, int sliding_window_len);
KWS* KWS_create_2(int16_t* audio_buffer);
KWS_free(KWS* kws);

void KWS_extract_features(KWS* kws);
void KWS_classify(KWS* kws);
void KWS_average_predictions(KWS* kws);
int KWS_get_top_class(KWS* kws, q7_t* prediction);

void KWS_init_kws(KWS* kws);
#endif
