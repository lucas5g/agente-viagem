from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Vou viajar para Londres em agosto de 2024, qual os melhores lugares para minha viagem?"},
  ]
)

print(response.choices[0].message.content)
