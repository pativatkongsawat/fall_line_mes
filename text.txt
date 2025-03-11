from fastapi import FastAPI
import os
import requests
from linebot import LineBotApi
from linebot.models import TextSendMessage
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("USER_ID")

if not ACCESS_TOKEN:
    raise ValueError("ACCESS_TOKEN ไม่ได้ตั้งค่าใน .env")

line_bot_api = LineBotApi(ACCESS_TOKEN)

app = FastAPI()

@app.post("/alert_fall")
def alert_fall():
    try:
        message = TextSendMessage(text="🚨 ตรวจพบการล้ม! กรุณาตรวจสอบทันที!")
        line_bot_api.push_message(USER_ID, message)
        return {"message": "✅ แจ้งเตือนการล้มสำเร็จ"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
