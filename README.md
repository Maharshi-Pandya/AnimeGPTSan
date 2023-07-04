# AnimeGPTSan

# AnimeGPTSan (Fine-tuned for Anime synopsis)

`AnimeGPTSan` is essentially the OpenAI's `GPT2` model fine-tuned on [this anime dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews). This model can be used to generate new descriptions/synopses for anime, based on some input prompt.

The original `GPT2` model can be found here: [OpenAI's GPT2 on HuggingFace](https://huggingface.co/gpt2)

# Results

- Input prompt

`The shadow realm`

- Outputs

1. `The shadow realm of the dark lord's hands will be invaded by the evil, demonic beasts. One day, the lord's minions, the Shattering Star God, awaken and attack the world...`
2. `The shadow realm of the Underworld awaits the demon lord Akito who is intent upon subjugating the human world. The Demon Lord is determined to destroy the world. He intends to use the "Dark Soul" as the catalyst...`
3. `The shadow realm is the source of many evil, but at its core, lies a magical weapon capable of killing any creature it touches. A group of teenagers in a high-tech school known as the Dark Knight Knights, nicknamed the "Darks," are a force that will...`
4. `The shadow realm has been awakened by the coming of the evil Demon King. The Shadow Warriors are a team of warriors who have been chosen to battle the Demon King with special powers...`
5. `The shadow realm exists between us and the demon world, and it is the source of the demons' power. The demons were sent to destroy the shadow realm, to make people mad. However, the evil spirit of the demon was able to...`

# Usage

The below code describes how to use `AnimeGPTSan` to generate anime synopsis given some input prompt.

> Note: Make sure to use a GPU for faster responses

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

anime_model = GPT2LMHeadModel.from_pretrained("maharshipandya/AnimeGPTSan")
anime_tokenizer = GPT2Tokenizer.from_pretrained("maharshipandya/AnimeGPTSan")

def generate_text(sequence):
  outputs = []
  ids = anime_tokenizer.encode(f"{sequence}", return_tensors="pt")
  final_outputs = anime_model.generate(ids, do_sample=True, max_length=200, top_k=40,
              top_p=0.95, temperature=1.0, num_return_sequences=10)

  for i, out in enumerate(final_outputs):
    output = anime_tokenizer.decode(out, skip_special_tokens=True)
    outputs.append(output)

  return outputs

print(generate_text("The shadow realm")) # list of generated synopses
```

In order to generate synopses without any prefix prompt, just give `"<|startoftext|>"` as the input prompt to `AnimeGPTSan`.