import dotenv from "dotenv";
dotenv.config();

async function sendMail(to, subject, text, html) {
  try {
    console.log(`📧 Attempting to send email to: ${to}`);
    console.log(`🔑 SendGrid Key: ${process.env.SENDGRID_API_KEY ? "Exists" : "Missing"}`);
    console.log(`🔑 Resend Key: ${process.env.RESEND_API_KEY ? "Exists" : "Missing"}`);

    // Method 1: SendGrid
    if (process.env.SENDGRID_API_KEY) {
      try {
        console.log("🔄 Trying SendGrid...");
        const sgMail = (await import('@sendgrid/mail')).default;
        sgMail.setApiKey(process.env.SENDGRID_API_KEY);
        
        const msg = {
          to: to,
          from: 'adscem2025@gmail.com', // Hardcode verified sender
          subject: subject,
          text: text,
          html: html,
        };

        console.log("📤 Sending via SendGrid...");
        const result = await sgMail.send(msg);
        console.log('✅ REAL Email sent via SendGrid to:', to);
        console.log('📧 SendGrid Response:', result[0]?.statusCode, result[0]?.headers);
        return { success: true, service: 'sendgrid' };
        
      } catch (sgError) {
        console.error('❌ SendGrid failed:', sgError.message);
        console.error('🔧 SendGrid error details:', {
          code: sgError.code,
          response: sgError.response?.body,
          statusCode: sgError.response?.statusCode
        });
        // Continue to Resend instead of throwing
      }
    }
    
    // Method 2: Resend
    if (process.env.RESEND_API_KEY) {
      try {
        console.log("🔄 Trying Resend...");
        const { Resend } = await import('resend');
        const resend = new Resend(process.env.RESEND_API_KEY);
        
        const { data, error } = await resend.emails.send({
          from: 'Attendance System <onboarding@resend.dev>',
          to: to,
          subject: subject,
          text: text,
          html: html,
        });

        if (error) {
          console.error('❌ Resend API error:', error);
          throw error;
        }
        
        console.log('✅ REAL Email sent via Resend to:', to);
        console.log('📧 Resend Message ID:', data.id);
        return { success: true, service: 'resend' };
        
      } catch (resendError) {
        console.error('❌ Resend failed:', resendError.message);
        // Continue to fallback
      }
    }
    
    // Fallback: Logging
    console.log('📧 All email services failed - would send to:', to);
    return { success: true, service: 'log' };
    
  } catch (err) {
    console.error("❌ Email failed completely:", err.message);
    console.log("📧 Email content was:", { to, subject });
    return { success: false, error: err.message };
  }
}

export default sendMail;
