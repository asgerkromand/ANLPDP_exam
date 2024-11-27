### Notes

The T5 encoder-decoder might arguably be better for question answering than encoder-models of similar size, as only the decoder part of this model is causally masked
and the encoder thus has access to all tokens in the input segment - in our case containing the context retrieved through RAG. 
Reference for masking: (Raffel et al., 2020: 18)

The same authors of T5 find that straightforward pre-training and fine-tuning yielded the best results, without resorting to 'gradual unfreezing' or
other strategies like that (Raffel et al., 2020: 30).
