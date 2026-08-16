# speaker_disentangled_prosody
Repository for training prosody encoders with speaker disentanglement.

## Setup

### Create conda environment
```bash
conda create --name speaker_disentangled_prosody python=3.7 -y
conda activate speaker_disentangled_prosody
```

### Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git --branch main --single-branch
cd ${cwd}/fairseq
git reset --hard 0b21875e45f332bedbcc0617dcf9379d3c03855f
pip install -e ./
```

### Install other requirements
```bash
pip install -r requirements.txt
```

## Data preparation

1. Download the [GigaSpeech dataset](https://github.com/SpeechColab/GigaSpeech)
2. Convert to HDF5 file, as feature extraction currently supports HDF5 file processing
3. Extract glottal source using `scripts/extract_glottal.py`
4. Extract raw prosody features to `GigaSpeech/feats` with `fairseq/examples/prosodyvec/simple_kmeans/learn_kmeans.py`
5. Normalize features with `fairseq/examples/prosodyvec/simple_kmeans/normalize_stats.py`
6. Train kmeans model and dump training/validation kmeans labels in `GigaSpeech/labels` (reference `fairseq/examples/hubert/simple_kmeans`)

### Training 

1. Copy `prosodyvec` directory to `fairseq` installation using `scripts/cp_prosodyvec_to_fairseq.sh`
2. Use `run_scripts/run_prosodyvec_gigaspeech_pretrain_single_glottal_lpf1000_normalized_rawprosody_spectraltilt_targets_spkr_adv_spanloss_wt1en1_lr_5em5_maskprob_0p5_masklen_8.sh`