#include <string.h>
#include "arm_math.h"
#include "mfcc.h"
#include "float.h"

// Define fixed-point type (e.g., q15_t for 16-bit fixed-point)
typedef q15_t fft_data_t;

MFCC* MFCC_create(int num_mfcc_features, int frame_len, int mfcc_dec_bits) 
{
    MFCC* mfcc = malloc(sizeof(MFCC));

    mfcc->num_mfcc_features = num_mfcc_features;
    mfcc->frame_len = frame_len;
    mfcc->mfcc_dec_bits = mfcc_dec_bits;

    // Round-up to nearest power of 2.
    mfcc->frame_len_padded = pow(2, ceil((log(frame_len)/log(2))));

    mfcc->frame = malloc(sizeof(fft_data_t)*mfcc->frame_len_padded);
    mfcc->buffer = malloc(sizeof(fft_data_t)*mfcc->frame_len_padded);
    mfcc->mel_energies = malloc(sizeof(float)*NUM_FBANK_BINS);

    // Create window function
    mfcc->window_func = malloc(sizeof(fft_data_t)*frame_len);
    for (int i = 0; i < frame_len; i++) {
        float win_val = 0.5 - 0.5 * cos(M_2PI * ((float)i) / (frame_len));
        mfcc->window_func[i] = (fft_data_t)(win_val * 32768); // Scale to Q15
    }

    // Create mel filterbank
    mfcc->fbank_filter_first = malloc(sizeof(int32_t)*NUM_FBANK_BINS);
    mfcc->fbank_filter_last = malloc(sizeof(int32_t)*NUM_FBANK_BINS);
    mfcc->mel_fbank = MFCC_create_mel_fbank(mfcc);

    // Create DCT matrix
    mfcc->dct_matrix = MFCC_create_dct_matrix(mfcc, NUM_FBANK_BINS, num_mfcc_features);

    // Initialize fixed-point FFT
    mfcc->rfft = malloc(sizeof(arm_rfft_fast_instance_q15));
    arm_rfft_fast_init_q15(mfcc->rfft, mfcc->frame_len_padded);

    return mfcc;
}

void MFCC_free(MFCC* mfcc) {
    free(mfcc->frame);
    free(mfcc->buffer);
    free(mfcc->mel_energies);
    free(mfcc->window_func);
    free(mfcc->fbank_filter_first);
    free(mfcc->fbank_filter_last);
    free(mfcc->dct_matrix);
    free(mfcc->rfft);

    for (int i = 0; i < NUM_FBANK_BINS; i++)
        free(mfcc->mel_fbank[i]);

    free(mfcc->mel_fbank);
    free(mfcc);
}

float * MFCC_create_dct_matrix(MFCC* mfcc, int32_t input_length, int32_t coefficient_count) {
    int32_t k, n;
    float * M = malloc(sizeof(float)*input_length*coefficient_count);
    float normalizer;
    arm_sqrt_f32(2.0/(float)input_length,&normalizer);
    for (k = 0; k < coefficient_count; k++) {
        for (n = 0; n < input_length; n++) {
            M[k*input_length+n] = normalizer * cos( ((double)M_PI)/input_length * (n + 0.5) * k );
        }
    }
    return M;
}

float ** MFCC_create_mel_fbank(MFCC* mfcc) {
    int32_t bin, i;

    int32_t num_fft_bins = mfcc->frame_len_padded/2;
    float fft_bin_width = ((float)SAMP_FREQ) / mfcc->frame_len_padded;
    float mel_low_freq = MelScale(MEL_LOW_FREQ);
    float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

    float *this_bin = malloc(sizeof(float)*num_fft_bins);
    float ** mel_fbank =  malloc(sizeof(float*)*NUM_FBANK_BINS);

    for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
        float left_mel = mel_low_freq + bin * mel_freq_delta;
        float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
        float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

        int32_t first_index = -1, last_index = -1;

        for (i = 0; i < num_fft_bins; i++) {
            float freq = (fft_bin_width * i);  // Center freq of this FFT bin.
            float mel = MelScale(freq);
            this_bin[i] = 0.0;

            if (mel > left_mel && mel < right_mel) {
                float weight;
                if (mel <= center_mel) {
                    weight = (mel - left_mel) / (center_mel - left_mel);
                } else {
                    weight = (right_mel - mel) / (right_mel - center_mel);
                }
                this_bin[i] = weight;
                if (first_index == -1)
                    first_index = i;
                last_index = i;
            }
        }

        mfcc->fbank_filter_first[bin] = first_index;
        mfcc->fbank_filter_last[bin] = last_index;
        mel_fbank[bin] = malloc(sizeof(float)*(last_index - first_index + 1)); 

        int32_t j = 0;
        // Copy the part we care about
        for (i = first_index; i <= last_index; i++) {
            mel_fbank[bin][j++] = this_bin[i];
        }
    }
    free(this_bin);
    return mel_fbank;
}

void MFCC_mfcc_compute(MFCC* mfcc, const int16_t * audio_data, q7_t* mfcc_out) {
    int32_t i, j, bin;

    // Normalize and convert .wav data to fixed-point
    for (i = 0; i < mfcc->frame_len; i++) {
        mfcc->frame[i] = (fft_data_t)((float)audio_data[i]); // Convert to Q15  ///(1<<15) * 32768
    }

    // Fill remaining with zeros
    memset(&mfcc->frame[mfcc->frame_len], 0, sizeof(fft_data_t) * (mfcc->frame_len_padded - mfcc->frame_len));

    // Apply window function
    for (i = 0; i < mfcc->frame_len; i++) {
        mfcc->frame[i] = (fft_data_t)(((int32_t)mfcc->frame[i] * (int32_t)mfcc->window_func[i]) >> 15); // Q15 multiplication
    }

    // Compute FFT
    arm_rfft_fast_q15(mfcc->rfft, mfcc->frame, mfcc->buffer, 0);

    // Convert to power spectrum
    int32_t half_dim = mfcc->frame_len_padded / 2;
    for (i = 0; i < half_dim; i++) {
        int32_t real = mfcc->buffer[i*2], im = mfcc->buffer[i*2 + 1];
        mfcc->buffer[i] = (real * real + im * im) >> 15; // Power calculation adjusted for Q15
    }

    float sqrt_data;
    // Apply mel filterbanks
    for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
        j = 0;
        float mel_energy = 0;
        int32_t first_index = mfcc->fbank_filter_first[bin];
        int32_t last_index = mfcc->fbank_filter_last[bin];
        for (i = first_index; i <= last_index; i++) {
            arm_sqrt_f32((float)mfcc->buffer[i],&sqrt_data); // Convert buffer to float before sqrt
            mel_energy += sqrt_data * mfcc->mel_fbank[bin][j++];
        }
        mfcc->mel_energies[bin] = mel_energy;

        // Avoid log of zero
        if (mel_energy == 0.0f)
            mfcc->mel_energies[bin] = FLT_MIN;
    }

    // Take log
    for (bin = 0; bin < NUM_FBANK_BINS; bin++)
        mfcc->mel_energies[bin] = logf(mfcc->mel_energies[bin]);

    // Take DCT. Uses matrix mul.
    for (i = 0; i < mfcc->num_mfcc_features; i++) {
        float sum = 0.0f;
        for (j = 0; j < NUM_FBANK_BINS; j++) {
            sum += mfcc->dct_matrix[i*NUM_FBANK_BINS + j] * mfcc->mel_energies[j];
        }

        // Input is Qx.mfcc_dec_bits (from quantization step)
        sum *= (0x1 << mfcc->mfcc_dec_bits);
        sum = roundf(sum); 
        if(sum >= 127)
            mfcc_out[i] = 127;
        else if(sum <= -128)
            mfcc_out[i] = -128;
        else
            mfcc_out[i] = (q7_t)sum; 
    }
}