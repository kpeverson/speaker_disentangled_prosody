#!/bin/bash

# change if necessary:
prosodyvec_dir=./prosodyvec
fairseq_dir=./fairseq/fairseq

rsync -a $prosodyvec_dir/ $fairseq_dir/
