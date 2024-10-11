import ollama



prompt1 = "Is this sarcasm: Hello, how are you, jackass? - Please answer with only one word (Yes/No). DO NOT write the reason"

prompt2 = "Is this sarcasm: You are a true dumbass! - Please answer with only one word (Yes/No). DO NOT write the reason"

list_of_prompts = [prompt1, prompt2]

for prompt in list_of_prompts:

    response = ollama.chat(model='phi3', messages=[
      {
        'role': 'user',
        'content': prompt,
      }, 
    ], options= {  # new
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048 # must be set, otherwise slightly random output
        })
    # print(response)
    print(response['message']['content'])

