from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage


load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("SECRET_TOKEN")
USER_ID = os.getenv("USER_ID")  


if not ACCESS_TOKEN or not CHANNEL_SECRET:
    raise ValueError("ACCESS_TOKEN หรือ CHANNEL_SECRET ไม่ได้ตั้งค่าใน .env")

line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

app = FastAPI()


@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        return {"error": str(e)}

    return {"message": "OK"}


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    message_text = event.message.text

    print(f"💬 Message: {message_text}")
    print(f"📌 User ID: {user_id}")
    

    
    if message_text.lower() in ["ขอ user id", "user id"]:
        reply_text = f"✅ User ID ของคุณคือ: {user_id}"
    else:
        reply_text = f"คุณส่งข้อความ: {message_text}"

    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@app.post("/alert_fall")
def alert_fall(video_url: str = "http://localhost:8000/videos/video_99.avi"):
    try:
        if not USER_ID:
            return {"error": "USER_ID ไม่ถูกตั้งค่าใน .env"}

        message = TextSendMessage(text=f"🚨 ตรวจพบการล้ม! กรุณาตรวจสอบทันที!\n🔗 วิดีโอ: {video_url}")
        line_bot_api.push_message(USER_ID, message)

        return {"message": "✅ แจ้งเตือนการล้มสำเร็จ", "sent_to": USER_ID, "video_url": video_url}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
