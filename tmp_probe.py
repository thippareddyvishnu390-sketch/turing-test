import os
import sys

os.chdir(r'C:\Users\AJAYKUMAR BANDI\turing-test')
import dotenv

dotenv.load_dotenv()

from google import genai
from google.genai import types

models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-flash-preview-05-20']
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)
cfg = types.GenerateContentConfig(temperature=0.2, candidate_count=1, max_output_tokens=180)
prompt = 'Reply with one short sentence.'

for m in models:
    try:
        resp = client.models.generate_content(
            model=m,
            contents=[types.Content(role='user', parts=[types.Part.from_text(text=prompt)])],
            config=cfg,
        )
        print('MODEL', m, 'OK', getattr(resp, 'text', None))
    except Exception as e:
        print('MODEL', m, 'ERR', type(e).__name__, e)
