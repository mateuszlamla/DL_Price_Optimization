# check_models.py
import google.generativeai as genai

api_key = 'AIzaSyARJX5trehnja0-Q31kKtFKePJUasdsGhI'

try:
    genai.configure(api_key=api_key)
    print("Szukam dostępnych modeli...\n")

    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            available_models.append(m.name)

    if not available_models:
        print("\nNie znaleziono modeli. Sprawdź czy Twoje API Key ma uprawnienia 'Generative Language API'.")
    else:
        print("\nSUKCES! Wybierz jedną z powyższych nazw (bez 'models/') i wpisz do app.py")

except Exception as e:
    print(f"Wystąpił błąd: {e}")