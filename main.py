from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models import GenerationRequest

app = FastAPI()

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained(
    "sberbank-ai/rugpt3medium_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained(
    "sberbank-ai/rugpt3medium_based_on_gpt2")


# http://127.0.0.1:1234/v1/chat/completions
@app.post("/v1/chat/completions")
async def generate_text(request: GenerationRequest):
    # Получение текста запроса пользователя из модели GenerationRequest
    user_request = request.messages[0].content

    # Преобразование входной строки в токены
    input_ids = tokenizer.encode(user_request, return_tensors="pt")

    # Генерация текста с использованием модели
    output = model.generate(input_ids, max_length=request.max_tokens,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            temperature=request.temperature,
                            do_sample=True)

    # Декодирование сгенерированного текста
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    response = {
        "completions": [
            {
                "id": "default",
                "model": "rugpt3medium_based_on_gpt2",
                "choices": [
                    {
                        "text": generated_text,
                        "finish_reason": "length"
                    }
                ]
            }
        ]
    }

    # Возвращение аналогичного ответа
    return response
