#!/usr/bin/env bash

#for d in train val;do
d=train
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku/tags-$d.vocab  $1/recom_text.tags.$d.filter.seg
python opennmt/bin/build_vocab.py --emb_vocab /home/hongmin/recom_nlg/emb/emb.vocab --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku/words-outs-$d.vocab  $1/recom_text.words.$d.filter.seg $1/recom_text.outs.$d.seg.temp
#python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku/words-$d.vocab  $1/recom_text.words.$d.filter
#done