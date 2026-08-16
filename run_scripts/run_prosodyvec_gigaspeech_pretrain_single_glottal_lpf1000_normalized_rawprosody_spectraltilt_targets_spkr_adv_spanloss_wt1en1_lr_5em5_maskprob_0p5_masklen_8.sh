#!/bin/bash

source /path/to/conda/bin/activate speaker_disentangled_prosody

prosodyvec_dir=/path/to/prosodyvec/dir
dataset_dir=${prosodyvec_dir}/GigaSpeech
glottal_hdf5_path=/path/to/glottal_segments.h5
expdir=${prosodyvec_dir}/exps/gigaspeech_from_hubert_glottal_lpf1000_normalized_rawprosody_spectraltilt_targets_spkr_adv_spanloss_wt1en1_lr_5em5_maskprob_0p5_masklen_8_large_batch_train_tmp
mkdir -p $expdir

# count number of speakers
utt2spkr_path=${dataset_dir}/metadata/utt2spkr.json
num_spkrs=$(grep -o '"[^"]*":[^,}]*' ${utt2spkr_path} | awk -F: '{print $2}' | tr -d '" ' | sort | uniq | wc -l)
echo "Number of speakers: $num_spkrs"

HYDRA_FULL_ERROR=1 python -u ${prosodyvec_dir}/fairseq/fairseq_cli/hydra_train.py  \
    --config-dir ${prosodyvec_dir}/prosodyvec/config/prosodyvec \
    --config-name prosodyvec_raw_prosody_gigaspeech_spkr_adv_spanloss_lr_5em5_manual_mask_large_batch.yaml \
    hydra.run.dir=${expdir} \
    task.data=${dataset_dir}/metadata \
    task.label_dir=${dataset_dir}/label/raw_prosody_feats_spectraltilt/speaker_normalized \
    task.labels=["km"] \
    task.crop=true \
    task.p_half_band=0.0 \
    task.spkr_adv_weight=0.1 \
    task.lpf_cutoff=1000.0 \
    task.utt2spkr_path=${dataset_dir}/metadata/utt2spkr.json \
    task.h5_path=${glottal_hdf5_path} \
    dataset.train_subset=training \
    dataset.valid_subset=validation \
    dataset.num_workers=10 \
    checkpoint.keep_best_checkpoints=1 \
    criterion.loss_weights=[10] \
    model.label_rate=62.5 \
    model.extractor_mode="default" \
    model.num_spkrs=$num_spkrs \
    model.spkr_class_stop_grad=false \
    model.spkr_class_layer_mode=final \
    model.mask_prob=0.5 \
    model.mask_length=8 \
    optimization.update_freq=[1] \
    optimization.max_update=2000000 \
    lr_scheduler.warmup_updates=8000 

