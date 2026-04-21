from groq import Groq


prompt = "Hello"

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a warm, empathetic mental health wellness buddy. Keep responses short and comforting."},
        {"role": "user", "content": prompt}
    ]
)

full_response = response.choices[0].message.content