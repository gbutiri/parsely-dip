"""Intent: Tell Weather

Returns current weather via OpenWeatherMap API.
"""

import os
import requests
from dotenv import load_dotenv
from parsely_dip.engine.registry import intent

load_dotenv(os.path.join(os.getenv('CLAUDE_PROJECT_DIR', '.'), '.env'))


def get_weather(city="Cleveland", lat=41.4993, lon=-81.6944):
    """Fetch weather from OpenWeatherMap API."""
    try:
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            return "Weather API key not set."
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_key}&units=imperial",
            timeout=5
        )
        data = response.json()
        if data.get("cod") != 200:
            return f"Couldn't get weather: {data.get('message', 'Unknown error')}"
        temp = round(data["main"]["temp"])
        description = data["weather"][0]["description"]
        return f"It's {temp}°F and {description} in {city}."
    except Exception as e:
        return f"Couldn't get weather: {str(e)}"


@intent('tell_weather')
def tell_weather():
    """Returns current weather."""
    return get_weather()
