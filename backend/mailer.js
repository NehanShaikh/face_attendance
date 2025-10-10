import dotenv from "dotenv";
dotenv.config();

async function sendMail(to, subject, text, html) {
  try {
    // Method 1: Resend (API-based - WORKS on Render free tier)
    if (process.env.RESEND_API_KEY) {
      const { Resend } = await import('resend');
      const resend = new Resend(process.env.RESEND_API_KEY);
      
      const { data, error } = await resend.emails.send({
        from: process.env.MAIL_FROM || 'Attendance System <onboarding@resend.dev>',
        to: to,
        subject: subject,
        text: text,
        html: html,
      });

      if (error) {
        console.error('❌ Resend error:', error);
        throw error;
      }
      
      console.log('✅ REAL Email sent via Resend to:', to);
      console.log('📧 Message ID:', data.id);
      return {
        success: true,
        messageId: data.id,
        service: 'resend'
      };
    }
    
    // Method 2: If no API key, show helpful error
    else {
      console.log('❌ No RESEND_API_KEY configured - emails will be logged only');
      console.log('📧 EMAIL WOULD BE SENT:');
      console.log('To:', to);
      console.log('Subject:', subject);
      console.log('Content:', text.substring(0, 100) + '...');
      
      // Return mock success so your app doesn't break
      return {
        success: true,
        messageId: 'logged-' + Date.now(),
        service: 'log',
        response: 'Email logged (no RESEND_API_KEY configured)'
      };
    }
    
  } catch (err) {
    console.error("❌ Email sending failed:", err.message);
    
    // Don't throw error - prevent breaking attendance/registration
    console.log("📧 Email content was:", { to, subject });
    
    return {
      success: false,
      error: err.message,
      service: 'error'
    };
  }
}

export default sendMail;
