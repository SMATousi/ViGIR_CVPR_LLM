import ollama

response = ollama.chat(model='phi3', messages=[
  {
    'role': 'user',
    'content': 'what did I just ask you?',
  },
])
print(response['message']['content'])
