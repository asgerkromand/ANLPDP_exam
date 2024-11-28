### Notes

The T5 encoder-decoder might arguably be better for question answering than encoder-models of similar size, as only the decoder part of this model is causally masked
and the encoder thus has access to all tokens in the input segment - in our case containing the context retrieved through RAG. 
Reference for masking: (Raffel et al., 2020: 18)

The same authors of T5 find that straightforward pre-training and fine-tuning yielded the best results, without resorting to 'gradual unfreezing' or
other strategies like that (Raffel et al., 2020: 30). Adapter layers (only fine-tuning additional RELU-blocks that are added after each pre-existing feedforward blocks in each block of the transformer) don't seem to work as well as just fine-tuning the whole thing either (ibid: 29). Unsupervised pre-training and fine-tuning also performs well compared to other methods mixing in multi-task-training+fine-tuning and stuff like that (ibid: 34).

Could make sense to use both BLEU and ROUGE for performance evaluation if possible. The T5 is likely to generate answers that are longer than the test ones we create, and may contain a lot of the n-grams in those long answers, thus generating quite high precision. ROUGE could shed light on how much of the stuff in the correct answer is left out with emphasis on the recall.

Perhaps we should reduce the precision of the floating points to FP16 to facilitate the fine-tuning.
