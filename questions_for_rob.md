# Questions

- Spørg om Readme. Nævn og scripts og yaml
- EOS-token == ' Spørgsmål', er det no-no?
- Vi har opdaget at de har identiske prompts bortset fra at der er newlines og 
ikke spaces i den til T5. De skal vel være identiske for sammenlignelighed?
- Formler
  - Udover ved retrieval, er der så nogle der er oplagte
  - F.eks. rogue, bleu.
- Modeller
  - Hvor meget skal vi gå ind i modellerne
  - Vores projekt centrerer RAG som ikke direkte er pensum, så hvordan sørger vi for, at få nok pensum med.
- What is a page on ITU?
- USE LLMs in coding regarding ACL policy.
- Visualiserig af vores metode. Har I nogle default softwares/pakker til at lave figurer til metodeafsnittet
- Hvordan oversætter vi bedst det kvantitative og kvalitative til vores projekt?
- Group contributions:
  - Hvordan deler man det op på ITU?
  - Er der en mulighed for at angive at man har lavet alt sammen.


# Takeaways

- Use model with the correct paragraphs as upper bound
- Check if the cls-token has actually been trained for DanskBERT (usually with next sentence prediction, could be that they masked it)
- We need to motivate why we're using the greedy approach, computation and legal stuff
