#!/usr/bin/env bash

#for d in train val;do
d=train
DOMAIN=sports
mkdir -p data/recom_nlg/$DOMAIN/one_per_sku/
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/$DOMAIN/one_per_sku/tags-$d.vocab /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.tags.filter.seg_new.$d
python opennmt/bin/build_vocab.py --emb_vocab /home/hongmin/recom_nlg/emb/emb.vocab --min_frequency 5 --save_vocab data/recom_nlg/$DOMAIN/one_per_sku/words-outs-$d.vocab /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.words.filter.seg_new.$d /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.outs.seg_new.full.$d
python opennmt/bin/build_vocab.py --emb_vocab /home/hongmin/recom_nlg/emb/emb.vocab --min_frequency 5 --save_vocab data/recom_nlg/$DOMAIN/one_per_sku/outs-$d.vocab /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.outs.seg_new.full.$d
python opennmt/bin/build_vocab.py --min_frequency 5 --save_vocab data/recom_nlg/$DOMAIN/one_per_sku/words-$d.vocab /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.words.filter.seg_new.$d /home/hongmin/recom_nlg/onmt/$DOMAIN/one_per_sku/seg_new/$d/recom_text_pos_all.txt.$DOMAIN.condense.json.int.trim_attr.all.outs.seg_new.attr_one.$d