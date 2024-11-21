# Legal Danish QA

Bedste modeller (snakmodel ser umiddelbart bedst ud nu)

- https://huggingface.co/NLPnorth/snakmodel-7b-base (modellen fylder ca. 12 GB i alt fordelt på 3 shards)
- https://huggingface.co/LumiOpen/Viking-7B
- https://huggingface.co/KennethTM/gpt-neo-1.3B-danish
Trænet på den danske del af OSCAR-datasættet (800 M danske tokens)
Ikke trænet på dansk fra bunden, på en eller anden måde overført med danske embeddings fra en engelsk model

# Andre modeller:

- https://huggingface.co/KennethTM/gpt2-medium-danish
- https://huggingface.co/Maltehb/aelaectra-danish-electra-small-cased
-Tror den her mest er til named entity recognition og sådan noget, ved ikke lige om den kan bruges til generation


# Datasæt

- https://gigaword.dk
- En stor del (næsten en fjerdedel vist) af datasættet er juridiske tekster, love og sådan noget fra internettet

# Evt.

Evt. BERT til encoding - https://huggingface.co/KennethTM
Har samme gut som tidligere også lavet
