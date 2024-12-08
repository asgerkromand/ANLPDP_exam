### Notes for 9/12 Adam

Jeg har prøvet at teste neo, men den tenderer til at gentage sig selv i det den genererer. Jeg har prøvet at forkorte det først ved at sætte eos-token til punktum,
men det sluttede bare for tidligt f.eks. med ord som f.eks. Jeg har også prøvet at sætte /n som eos-token da modellen typisk laver newline inden den gentager
sig selv. Det kan godt være at vi bare bliver nødt til at lade den generere det hele selvom det tager mere compute og tid, og så splitte outputs manuelt
bagefter ved newlines for at få de korrekte output ud. 
