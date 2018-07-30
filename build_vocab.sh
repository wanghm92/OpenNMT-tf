#!/usr/bin/env bash

#for d in train val;do
d=train
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/multi_per_sku/tags-$d.vocab  $1/recom_text.tags.$d
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/multi_per_sku/words-outs-$d.vocab  $1/recom_text.words.$d $1/recom_text.outs.$d
#done