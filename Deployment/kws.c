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


#include "kws.h"

KWS* KWS_create_1(int record_win, int sliding_win_len)
{
  KWS* kws = malloc(sizeof(KWS));
  kws->nn = DS_CNN_create();
  kws->recording_win = record_win;
  kws->sliding_window_len = sliding_win_len;
  KWS_init_kws(kws);

  return kws;
}

KWS* KWS_create_2(int16_t* audio_data_buffer)
{
  KWS* kws = malloc(sizeof(KWS));
  kws->nn = DS_CNN_create();
  kws->audio_buffer = audio_data_buffer;
  kws->recording_win = kws->nn->num_frames;
  kws->sliding_window_len = 1;
  KWS_init_kws(kws);

  return kws;
}

KWS_free(KWS* kws)
{
  DS_CNN_free(kws->nn);
 
  MFCC_free(kws->mfcc);
  free(kws->mfcc_buffer);
  free(kws->output);
  free(kws->predictions);
  free(kws->averaged_output);

  free(kws);
}

void KWS_init_kws(KWS* kws)
{
  kws->num_mfcc_features = kws->nn->num_mfcc_features;
  kws->num_frames = kws->nn->num_frames;
  kws->frame_len = kws->nn->frame_len;
  kws->frame_shift = kws->nn->frame_shift;
  int mfcc_dec_bits = kws->nn->in_dec_bits;
  kws->num_out_classes = kws->nn->num_out_classes;
  kws->mfcc = MFCC_create(kws->num_mfcc_features, kws->frame_len, mfcc_dec_bits);
  kws->mfcc_buffer = malloc(sizeof(q7_t)*kws->num_frames*kws->num_mfcc_features);
  kws->output =  malloc(sizeof(q7_t)*kws->num_out_classes);
  kws->averaged_output =  malloc(sizeof(q7_t)*kws->num_out_classes);
  kws->predictions =  malloc(sizeof(q7_t)*kws->sliding_window_len*kws->num_out_classes);
  kws->audio_block_size = kws->recording_win*kws->frame_shift;
  kws->audio_buffer_size = kws->audio_block_size + kws->frame_len - kws->frame_shift;
}

void KWS_extract_features(KWS* kws) 
{
  if(kws->num_frames>kws->recording_win) {
    //move old features left 
    memmove(kws->mfcc_buffer, kws->mfcc_buffer+(kws->recording_win*kws->num_mfcc_features),
        (kws->num_frames-kws->recording_win)*kws->num_mfcc_features);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (kws->num_frames-kws->recording_win)*kws->num_mfcc_features; 
  for (uint16_t f = 0; f < kws->recording_win; f++) {
    MFCC_mfcc_compute(kws->mfcc, kws->audio_buffer+(f*kws->frame_shift),&kws->mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += kws->num_mfcc_features;
  }
}

void KWS_classify(KWS* kws)
{
  DS_CNN_run_nn(kws->nn, kws->mfcc_buffer, kws->output);
  // Softmax
  arm_softmax_q7(kws->output,kws->num_out_classes,kws->output);
}

int KWS_get_top_class(KWS* kws, q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<kws->num_out_classes;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS_average_predictions(KWS* kws)
{
  // shift the old predictions left
  arm_copy_q7((q7_t *)(kws->predictions+kws->num_out_classes), (q7_t *)kws->predictions, 
        (kws->sliding_window_len-1)*kws->num_out_classes);
  // add new predictions at the end
  arm_copy_q7((q7_t *)kws->output, (q7_t *)(kws->predictions+(kws->sliding_window_len-1)*kws->num_out_classes), 
        kws->num_out_classes);
  //compute averages
  int sum;
  for(int j=0;j<kws->num_out_classes;j++) {
    sum=0;
    for(int i=0;i<kws->sliding_window_len;i++) 
      sum += kws->predictions[i*kws->num_out_classes+j];

    kws->averaged_output[j] = (q7_t)(sum/kws->sliding_window_len);
  }   
}
  

#if 1
int MFCC_test(uint8_t* in_audio_buffer)
{
  const char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
  KWS* kws = KWS_create_2(in_audio_buffer);

  //T.start();
  //int start=T.read_us();
  KWS_extract_features(kws); //extract mfcc features
  KWS_classify(kws);	  //classify using dnn
  //int end=T.read_us();
  //T.stop();
  int max_ind = KWS_get_top_class(kws, kws->output);
  //pc.printf("Total time : %d us\r\n",end-start);
  printf("Detected %s (%d%%)\r\n",output_class[max_ind],((int)kws->output[max_ind]*100/128));

  return max_ind;
}
#endif