Title:

PolyCMD: A Reproducible Multilingual Lexicon for Grounding Robot Command Words in Unseen Languages and Scripts

Abstract:

Service robots that work with speakers of many languages must understand the short words that parameterise commands, such as numbers, units, weekdays and months, including words from languages and writing systems that were absent from training. We present two openly available lexical resources for this task. PolyCMD contains 1,813 command words for 43 concepts in 44 languages, 16 genealogical groups and 13 scripts, and PolyCMD-U contains 424 words in 11 languages that are absent from the pretraining data of current multilingual encoders, including Cherokee and Chakma, whose scripts do not occur in PolyCMD. Both lexicons are generated from the Unicode Common Locale Data Repository by released scripts, documented with a data statement, and can be rebuilt exactly. On these resources we evaluate grounding of a held-out language with string-similarity baselines, three pretrained encoders and PolyNET, a small reference model with 34.5 thousand parameters that romanises every script, fuses three similarity memories with a NETtalk-style letter-window network and decodes a session jointly under mutual exclusivity. In leave-one-language-out evaluation on 36 unseen languages PolyNET grounds 81.8% of the words, against 56.3% for a NETtalk network on raw Unicode. With the whole language family withheld it is significantly better than every similarity baseline (67.4% against 63.5%, Holm-corrected p = 4.40 × 10⁻³). Because transfer depends on shared vocabulary, a pretrained encoder that has seen all test languages reaches 95.6%, while on PolyCMD-U PolyNET grounds 55.1%, from 100% for Chakma to 5% for Cherokee.

Keywords:

Multilingual lexicon,
Language resources for robotics,
Reproducible benchmark,
Zero-shot cross-lingual transfer,
Low-resource languages,
Transliteration.
