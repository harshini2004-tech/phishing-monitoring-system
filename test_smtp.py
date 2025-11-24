#!/usr/bin/env python3
import smtplib
from email.mime.text import MIMEText

def test_gmail_smtp():
    try:
        # Gmail SMTP settings
        smtp_server = "smtp.gmail.com"
        port = 587
        sender_email = "harshininmurthy@gmail.com"
        # Use your 16-character app password here
        password = "cheh xlms dxnf kqck"
        
        # Create message
        message = MIMEText("This is a test email from your phishing detection system!")
        message["Subject"] = "Test Alert from Phishing Detection"
        message["From"] = sender_email
        message["To"] = sender_email
        
        # Send email
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, sender_email, message.as_string())
        server.quit()
        
        print("✅ Test email sent successfully!")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

if __name__ == "__main__":
    test_gmail_smtp()
