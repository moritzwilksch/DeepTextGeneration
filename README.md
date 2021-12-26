# Deep Text Generation 📝
> Experimenting with small deep learning models for natural text generation

## ToDo
- [x] implement word-based model
- [x] implement char-based model
- [x] try exponential LR decay 
- [ ] try subword tokenizer
- [x] implement temperature
- [ ] try [keras mini GPT example](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)
- [x] use [German recipes](https://www.kaggle.com/sterby/german-recipes-dataset) data set for more standardized corpus
- [x] benchmark mixed precision: `tf.keras.mixed_precision.set_global_policy("mixed_float16")`
- [x] implement shifted-sequence model like [this](https://www.tensorflow.org/text/tutorials/text_generation) rather than manipulating data set 

# Data 💿
I'm training on a corpus of German recipes scraped from `chefkoch.de` and available on [kaggle](https://www.kaggle.com/sterby/german-recipes-dataset)

# Models 🤖
WIP.

# Results 🔬

## Examples

| 🇩🇪 | 🇺🇸 |
|---|---|
|**zuerst hähnchen**brustfilet in scheiben schneiden. die masse dazugeben, dann die zwiebel in ein großen topf geben. salzen und pfeffern. mit den geflügelbrühe auffüllen und einkochen lassen. die zwiebeln abziehen und klein schneiden. die salamatkerlisieren. das wasser und fischscheiben in den topf geben und alles aufkochen lassen. auf einem sieb abtropfen lassen. | First cut the chicken breast fillet into slices. add the mixture, then put the onion in a large saucepan. salt and pepper. top up with the poultry stock and let it boil down. Peel and chop the onions. the salamat kerlize. add the water and fish slices to the pan and bring to the boil. drain on a sieve. |


# Take-Aways 💡
- character-based model outperforms word-based model with similar training time (char-based model has 1/10th of the parameters)
  - word-based accuracy: ~50%, no coherent sentences.
  - char-based accuracy: ~80%, mostly coherent text
- the model only works as a real RNN Seq2Seq model ([link](https://www.tensorflow.org/text/tutorials/text_generation)), not when manipulating the data set to simulate an expanding window sequence.
- mixed precision training yields a `45%` speed-up with only a slight reduction in accuracy (77% vs 76% after 20 epochs) on a NVIDIA Tesla T4 GPU
