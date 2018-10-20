#!/usr/bin/env bash

#for d in train val;do
d=train
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku_dedup/tags-$d.vocab  $1/recom_text.tags.$d.filter.seg_new # $1/recom_text.outs.$d.seg_new.tag_one
python opennmt/bin/build_vocab.py --emb_vocab /home/hongmin/recom_nlg/emb/emb.vocab --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku_dedup/words-outs-$d.vocab $1/recom_text.words.$d.filter.seg_new $1/recom_text.outs.$d.seg_new.attr_one $1/recom_text.outs.$d.seg_new.temp
python opennmt/bin/build_vocab.py --emb_vocab /home/hongmin/recom_nlg/emb/emb.vocab --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku_dedup/outs-$d.vocab $1/recom_text.outs.$d.seg_new.temp
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/clothes/one_per_sku_dedup/words-$d.vocab  $1/recom_text.words.$d.filter.seg_new $1/recom_text.outs.$d.seg_new.attr_one
python opennmt/bin/build_vocab.py --min_frequency 50 --save_vocab data/recom_nlg/clothes/one_per_sku_dedup/words-filter.vocab  $1/recom_text.words.$d.filter.seg_new $1/recom_text.outs.$d.seg_new.attr_one
#done