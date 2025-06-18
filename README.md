
# Keyword spotting for Microcontrollers 

## test 
main()
{
    void* test_pcm = wav_arrary;
    MFCC_test(test_pcm);
}



## train 

#############
modify common.sh 

MODEL_SIZE_INFO="5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1"
DCT_CNT=10
WIN_SIZE=20
WIN_STRIDE=10

then,

sh train.sh 
or
python train.py       --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20     --summaries_dir work/DS_CNN/logs --train_dir work/DS_CNN/training

sh test.sh 
or
python test.py    --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20   --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000

sh freeze.sh 
or
python freeze.py    --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20   --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000 --output_file work/DS_CNN/ds_cnn.pb


sh label_wav.sh 
or
python label_wav.py --wav left.wav --graph work/DS_CNN/ds_cnn.pb --labels work/DS_CNN/training/ds_cnn_labels.txt --how_many_labels 1
python label_wav.py --wav test_yes.wav --graph work/DS_CNN/ds_cnn.pb --labels work/DS_CNN/training/ds_cnn_labels.txt --how_many_labels 1

## quant

sh fold_batchnorm.sh 
or
python fold_batchnorm.py    --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20     --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000

sh quant_test.sh 
or
 python quant_test.py       --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20     --act_max 0 0 0 0 0 0 0 0 0 0 0 0  --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000_bnfused


<!-- python quant_dump.py       --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20     --act_max 0 0 0 0 0 0 0 0 0 0 0 0  --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000_bnfused -->

#test  32, 4, 16, 8, 8, 4, 8, 8, 32, 32, 4, 8
sh quant_dump.sh 
or
python quant_dump.py       --data_url=   --data_dir=../speech_data   --wanted_words yes,no,up,down,left,right,on,off,stop,go   --clip_duration_ms 1000   --model_architecture ds_cnn   --testing_percentage 10   --validation_percentage 10   --training_percentage 80   --unknown_percentage 50      --model_size_info 5 32 10 4 2 2 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1 32 3 3 1 1   --dct_coefficient_count 10   --window_size_ms 40   --window_stride_ms 20     --act_max 32 4 16 8 8 4 8 8 32 32 4 8  --checkpoint work/DS_CNN/training/best/ds_cnn_9338.ckpt-8000_bnfused

hill generate c code

## C deployment
copy ds_cnn.h ds_cnn.h ds_cnn_weights.h to project.


## Test



