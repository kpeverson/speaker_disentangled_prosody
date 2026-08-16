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

### Download the [GigaSpeech dataset](https://github.com/SpeechColab/GigaSpeech)