import nodemailer from "nodemailer";
import dotenv from "dotenv";
dotenv.config();

let transporter;

if (process.env.EMAIL_USER && process.env.EMAIL_PASS) {
  transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS
    }
  });
  
  // Verify transporter connection
  transporter.verify(function (error, success) {
    if (error) {
      console.error('❌ SMTP Connection Error:', error);
    } else {
      console.log('✅ SMTP Server is ready to take our messages');
      console.log('📧 Using email:', process.env.EMAIL_USER);
      console.log('📧 From address:', process.env.MAIL_FROM || process.env.EMAIL_USER);
    }
  });
  
} else {
  transporter = nodemailer.createTransport({
    streamTransport: true,
    newline: "unix",
    buffer: true
  });
  console.warn("⚠️ Mailer running in simulated mode (no EMAIL_USER/EMAIL_PASS set). Mails will be logged, not sent.");
}

async function sendMail(to, subject, text, html) {
  try {
    console.log(`📧 Preparing to send email to: ${to}`);
    console.log(`📧 From address: ${process.env.MAIL_FROM || process.env.EMAIL_USER}`);
    
    const info = await transporter.sendMail({
      from: process.env.MAIL_FROM || process.env.EMAIL_USER, // ✅ FIXED: Use MAIL_FROM if available
      to,
      subject,
      text,
      html
    });

    if (info && info.message) {
      console.log("📧 (simulated) Mail content:\n", info.message.toString());
    } else {
      console.log("✅ Mail sent successfully!");
      console.log("📧 Message ID:", info.messageId);
      console.log("📧 Response:", info.response);
    }
    return info;
  } catch (err) {
    console.error("❌ Mail failed", err);
    throw err;
  }
}

export default sendMail;
