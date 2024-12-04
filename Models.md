# Legal Danish QA

Bedste modeller (dant5-large ser umiddelbart mest brugbar ud)

- https://huggingface.co/strombergnlp/dant5-large
- https://huggingface.co/NLPnorth/snakmodel-7b-base (modellen fylder ca. 12 GB i alt fordelt på 3 shards)
- https://huggingface.co/KennethTM/gpt-neo-1.3B-danish
Trænet på den danske del af OSCAR-datasættet (800 M danske tokens)
Ikke trænet på dansk fra bunden, på en eller anden måde overført med danske embeddings fra en engelsk model

# Andre modeller:

- https://huggingface.co/KennethTM/gpt2-medium-danish
- https://huggingface.co/mhenrichsen/danskgpt-tiny-chat
  (Ovenstående har vist 1.1 B parametre selvom den hedder tiny)
- https://huggingface.co/EleutherAI/gpt-neo-125m


# Datasæt

- https://gigaword.dk
- En stor del (næsten en fjerdedel vist) af datasættet er juridiske tekster, love og sådan noget fra internettet

# Evt.

Evt. BERT til encoding - https://huggingface.co/KennethTM
- https://huggingface.co/vesteinn/DanskBERT
Har samme gut som tidligere også lavet
