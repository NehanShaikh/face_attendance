import nodemailer from "nodemailer";
import dotenv from "dotenv";
dotenv.config(); // ensure .env is loaded when using this module

// If MAIL_USER / MAIL_PASS present -> use real SMTP (Gmail example)
// Otherwise use a safe "log only" transport for local dev.
let transporter;

if (process.env.EMAIL_USER && process.env.EMAIL_PASS) {
  transporter = nodemailer.createTransport({
    service: "gmail", // simpler than host/port
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS
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
    const info = await transporter.sendMail({
      from: process.env.MAIL_FROM || process.env.MAIL_USER || "no-reply@example.com",
      to,
      subject,
      text,
      html
    });

    // If streamTransport used, info.message contains the full message buffer
    if (info && info.message) {
      console.log("📧 (simulated) Mail content:\n", info.message.toString());
    } else {
      console.log("✅ Mail sent:", info && (info.response || info.messageId));
    }
  } catch (err) {
    console.error("❌ Mail failed", err);
  }
}

export default sendMail;

