import dotenv from "dotenv";
dotenv.config();

async function sendMail(to, subject, text, html) {
  try {
    // Method 1: SendGrid (free tier - works with any email)
    if (process.env.SENDGRID_API_KEY) {
      const sgMail = (await import('@sendgrid/mail')).default;
      sgMail.setApiKey(process.env.SENDGRID_API_KEY);
      
      const msg = {
        to: to,
        from: process.env.MAIL_FROM || 'adscem2025@gmail.com',
        subject: subject,
        text: text,
        html: html,
      };

      await sgMail.send(msg);
      console.log('✅ REAL Email sent via SendGrid to:', to);
      return { success: true, service: 'sendgrid' };
    }
    
    // Method 2: Resend (limited to verified emails)
    else if (process.env.RESEND_API_KEY) {
      const { Resend } = await import('resend');
      const resend = new Resend(process.env.RESEND_API_KEY);
      
      const { data, error } = await resend.emails.send({
        from: process.env.MAIL_FROM || 'Attendance System <onboarding@resend.dev>',
        to: to,
        subject: subject,
        text: text,
        html: html,
      });

      if (error) throw error;
      
      console.log('✅ REAL Email sent via Resend to:', to);
      return { success: true, service: 'resend' };
    }
    
    // Fallback: Logging
    else {
      console.log('📧 Email would be sent to:', to);
      return { success: true, service: 'log' };
    }
    
  } catch (err) {
    console.error("❌ Email failed:", err.message);
    console.log("📧 Email content was:", { to, subject });
    return { success: false, error: err.message };
  }
}

export default sendMail;
