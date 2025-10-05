import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import path from "path";
import { fileURLToPath } from "url";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import sequelize from "./db.js";
import { Pool } from "pg";
import sendMail from "./mailer.js"; // ✅ note the .js extension
import { exec } from "child_process";


const app = express();
const port = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET;

// Fix __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "../frontend")));

// Sequelize connection test
(async () => {
  try {
    await sequelize.authenticate();
    console.log("✅ PostgreSQL Connected via Sequelize!");
    await sequelize.sync();
    console.log("✅ Tables Synced!");
  } catch (err) {
    console.error("❌ DB connection failed:", err);
  }
})();

// Pool connection
const db = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

db.connect()
  .then(() => console.log("✅ PostgreSQL Pool Connected!"))
  .catch((err) => console.error("❌ Pool connection error:", err));

app.get("/", (req, res) => {
  res.send("Backend is running!");
});

// ---------------- AUTH MIDDLEWARE ----------------
// ---------------- AUTH MIDDLEWARE ----------------

// Generic token authentication
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    if (!authHeader) return res.sendStatus(401);

    const token = authHeader.split(' ')[1];
    if (!token) return res.sendStatus(401);

    jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) return res.sendStatus(403);
        req.user = decoded;
        next();
    });
}

// Student authentication
function authenticateStudent(req, res, next) {
    const authHeader = req.headers["authorization"];
    if (!authHeader) return res.status(401).json({ message: "No token provided" });

    const token = authHeader.split(" ")[1];
    if (!token) return res.status(401).json({ message: "Invalid token" });

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        if (decoded.role !== "student") {
            return res.status(403).json({ message: "Access denied: Not a student" });
        }
        req.user = decoded; // ✅ includes username
        next();
    } catch (err) {
        console.error(err);
        res.status(401).json({ message: "Invalid token" });
    }
}

// Faculty authentication
function authenticateFaculty(req, res, next) {
    const authHeader = req.headers["authorization"];
    if (!authHeader) return res.sendStatus(401);

    const token = authHeader.split(" ")[1];
    if (!token) return res.sendStatus(401);

    jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) return res.sendStatus(403);

        if (decoded.role !== "faculty") {
            return res.status(403).json({ error: "Access denied: Not a faculty" });
        }

        req.user = decoded;
        next();
    });
}

// Admin check
function isAdmin(req, res, next) {
    if (req.user && req.user.role === "admin") {
        return next();
    }
    return res.status(403).json({ error: "Access denied. Admins only." });
}

// ---------------- LOGIN ----------------
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  try {
    // Query user by username
    const result = await db.query("SELECT * FROM users WHERE username = $1", [username]);

    if (result.rows.length === 0) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const user = result.rows[0];
    const storedPassword = user.password;

    // Check if password is hashed (bcrypt)
    let passwordMatch = false;

    if (storedPassword.startsWith("$2b$")) {
      passwordMatch = await bcrypt.compare(password, storedPassword);
    } else {
      passwordMatch = password === storedPassword;
    }

    if (!passwordMatch) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    // Generate JWT token
    const token = jwt.sign(
      {
        user_id: user.user_id,
        username: user.username,   // ✅ include username
        role: user.role,
        student_id: user.student_id,
        faculty_id: user.faculty_id,
      },
      JWT_SECRET,
      { expiresIn: "1h" }
    );

    res.json({
      message: "Login successful",
      token,
      role: user.role,
      userId: user.user_id,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ message: "Internal server error" });
  }
});


// ================== Attendance Routes ==================

// Student marks their attendance (with subject check)


// ---------------- MARK ATTENDANCE ----------------
app.post("/attendance", async (req, res) => {
  try {
    const { name } = req.body;

    // 1. Find student info
    const studentResult = await db.query(
      "SELECT student_id, name, email FROM students WHERE name = $1",
      [name]
    );

    if (studentResult.rowCount === 0) {
      console.warn("Student not found:", name);
      return res.status(404).send("Student not found");
    }

    const studentId = studentResult.rows[0].student_id;
    const studentName = studentResult.rows[0].name;
    const studentEmail = studentResult.rows[0].email;

    // 2. Find current subject from timetable (FIXED TIMEZONE)
    const subjectResult = await db.query(
      `SELECT s.subject_id, s.name AS subject_name
       FROM timetable t
       JOIN subjects s ON t.subject_id = s.subject_id
       JOIN student_timetable st ON st.timetable_id = t.timetable_id
       WHERE st.student_id = $1
         AND LOWER(TRIM(t.day)) = LOWER(TO_CHAR(NOW() AT TIME ZONE 'Asia/Kolkata', 'FMDay'))
         AND (NOW() AT TIME ZONE 'Asia/Kolkata')::time BETWEEN t.start_time AND t.end_time
       LIMIT 1`,
      [studentId]
    );

    if (subjectResult.rowCount === 0) {
      console.warn(`No subject found for ${studentName} at this time`);
      return res.status(404).send("No subject at this time or student not enrolled");
    }

    const subjectId = subjectResult.rows[0].subject_id;
    const subjectName = subjectResult.rows[0].subject_name;

    // 3. Prevent duplicate attendance (FIXED TIMEZONE)
    const checkResult = await db.query(
      `SELECT 1 FROM attendance
       WHERE student_id = $1
         AND subject_id = $2
         AND DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = (NOW() AT TIME ZONE 'Asia/Kolkata')::date`,
      [studentId, subjectId]
    );

    if (checkResult.rowCount > 0) {
      console.log(`Attendance already marked for ${studentName} in ${subjectName}`);
      return res.status(400).send("Attendance already marked for this subject today");
    }

    // 4. Insert attendance
    await db.query(
      `INSERT INTO attendance (student_id, subject_id, subject_name)
       VALUES ($1, $2, $3)`,
      [studentId, subjectId, subjectName]
    );

    const timestamp = new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" });
    console.log(`✅ Attendance marked for ${studentName} (Subject: ${subjectName}) at ${timestamp}`);

    // 5. Send email if student has one
    if (studentEmail) {
      await sendMail(
        studentEmail,
        "Attendance Marked",
        `Your attendance for ${subjectName} has been marked.`,
        `<h3>Hello ${studentName},</h3>
         <p>Your attendance has been marked successfully.</p>
         <p><b>Subject:</b> ${subjectName}<br>
            <b>Time:</b> ${timestamp}</p>`
      );
    } else {
      console.warn(`No email for student ${studentName} — skipping mail.`);
    }

    res.send({
      status: "success",
      student_id: studentId,
      subject: subjectName,
      timestamp,
    });

  } catch (err) {
    console.error("Error in /attendance:", err);
    res.status(500).send("Internal server error");
  }
});

// Add this temporary test route
app.get("/debug-time", async (req, res) => {
  try {
    const timeDebug = await db.query(
      `SELECT 
         TO_CHAR(NOW() AT TIME ZONE 'Asia/Kolkata', 'FMDay') as current_day,
         (NOW() AT TIME ZONE 'Asia/Kolkata')::time as current_time,
         (NOW() AT TIME ZONE 'Asia/Kolkata') as full_timestamp`
    );
    
    // Also check timetable data
    const timetableData = await db.query(`
      SELECT s.name as student_name, t.day, t.start_time, t.end_time, sub.name as subject_name
      FROM student_timetable st
      JOIN students s ON st.student_id = s.student_id
      JOIN timetable t ON st.timetable_id = t.timetable_id
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE s.name = 'nehan'
    `);
    
    res.json({
      current_time: timeDebug.rows[0],
      timetable_entries: timetableData.rows
    });
  } catch (error) {
    console.error("Debug error:", error);
    res.status(500).json({ error: error.message });
  }
});

// ---------------- GET ALL ATTENDANCE ----------------
app.get("/attendance", async (req, res) => {
  try {
    const result = await db.query(
      `SELECT a.attendance_id,
              st.name AS student_name,
              a.subject_name,
              a.timestamp
       FROM attendance a
       JOIN students st ON a.student_id = st.student_id
       ORDER BY a.timestamp DESC`
    );
    res.json(result.rows);
  } catch (err) {
    console.error("❌ Error fetching attendance:", err);
    res.status(500).send("Internal Server Error");
  }
});

// ---------------- GET STUDENT ATTENDANCE ----------------
app.get("/student/attendance", authenticateToken, async (req, res) => {
  try {
    if (req.user.role !== "student") {
      return res.status(403).json({ error: "Forbidden" });
    }

    const result = await db.query(
      `SELECT a.attendance_id,
              st.student_id,
              st.name AS student_name,
              a.subject_name,
              a.timestamp
       FROM attendance a
       JOIN students st ON a.student_id = st.student_id
       WHERE st.student_id = $1
       ORDER BY a.timestamp DESC`,
      [req.user.student_id]
    );

    res.json(result.rows);
  } catch (err) {
    console.error("❌ Error fetching student attendance:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---------------- FACULTY ROUTES ----------------
// Get all students
// Get all students
app.get('/faculty/students', authenticateToken, async (req, res) => {
  if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

  try {
    const result = await db.query("SELECT * FROM students");
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Add student
app.post("/faculty/students", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const { name, roll_number, class_name, email, phone } = req.body;

  try {
    const insertResult = await db.query(
      "INSERT INTO students (name, roll, class, email, phone) VALUES ($1, $2, $3, $4, $5) RETURNING student_id",
      [name, roll_number, class_name, email, phone]
    );

    const newStudentId = insertResult.rows[0].student_id;

    // Send welcome email
    try {
      await sendMail(
        email,
        "🎓 Student Registration - Smart Attendance System",
        `Hello ${name},\n\nYou have been successfully registered by your faculty.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nYou can now use your account to mark attendance.\n\nBest Regards,\nFaculty Team`,
        `<h2>Welcome, ${name}! 🎓</h2>
         <p>Your faculty has registered you in the <b>Smart Attendance System</b>.</p>
         <p>
           <b>Roll Number:</b> ${roll_number}<br>
           <b>Class:</b> ${class_name}<br>
           <b>Student ID:</b> ${newStudentId}
         </p>
         <p>You can now use your account to mark attendance.</p>
         <br>
         <p>Best Regards,<br><b>Faculty Team</b></p>`
      );

      console.log(`📧 Registration email sent to ${email}`);
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    res.json({
      message: "Student added & email sent",
      student_id: newStudentId,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update student
app.put('/faculty/students/:id', authenticateToken, async (req, res) => {
  if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

  const id = req.params.id;
  const { name, roll_number, class_name, email, phone } = req.body;

  try {
    const result = await db.query(
      "UPDATE students SET name=$1, roll=$2, class=$3, email=$4, phone=$5 WHERE student_id=$6 RETURNING *",
      [name, roll_number, class_name, email, phone, id]
    );

    if (result.rows.length === 0) return res.status(404).json({ error: "Student not found" });
    res.json({ message: "Student updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete student
app.delete('/faculty/students/:id', authenticateToken, async (req, res) => {
  if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

  const id = req.params.id;

  try {
    const result = await db.query(
      "DELETE FROM students WHERE student_id=$1 RETURNING *",
      [id]
    );

    if (result.rows.length === 0) return res.status(404).json({ error: "Student not found" });
    res.json({ message: "Student deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});



// Get all students assigned to the faculty
app.get("/faculty/assigned-students", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const facultyId = req.user.faculty_id;

  try {
    const result = await db.query(`
      SELECT fs.id, s.student_id, s.name, s.roll_number, s.class, s.email, s.phone, fs.assigned_on
      FROM faculty_students fs
      JOIN students s ON fs.student_id = s.student_id
      WHERE fs.faculty_id = $1
    `, [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Assign a student to faculty
app.post("/faculty/assigned-students", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const facultyId = req.user.faculty_id;
  const { student_id } = req.body;

  try {
    await db.query(
      "INSERT INTO faculty_students (faculty_id, student_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
      [facultyId, student_id]
    );

    // Assign all timetable subjects of this faculty
    await db.query(`
      INSERT INTO student_timetable (student_id, timetable_id)
      SELECT $1, t.timetable_id
      FROM timetable t
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE sub.faculty_id = $2
      ON CONFLICT DO NOTHING
    `, [student_id, facultyId]);

    res.json({ message: "✅ Student assigned and timetable updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Unassign student from faculty
app.delete("/faculty/assigned-students/:studentId", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const facultyId = req.user.faculty_id;
  const studentId = req.params.studentId;

  try {
    const deleteResult = await db.query(
      "DELETE FROM faculty_students WHERE faculty_id = $1 AND student_id = $2 RETURNING *",
      [facultyId, studentId]
    );

    if (deleteResult.rowCount === 0) return res.status(404).json({ error: "Student not assigned to this faculty" });

    // Remove student from this faculty's subject timetables
    await db.query(`
      DELETE FROM student_timetable st
      USING timetable t
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE st.timetable_id = t.timetable_id AND st.student_id = $1 AND sub.faculty_id = $2
    `, [studentId, facultyId]);

    res.json({ message: "✅ Student unassigned and timetable updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Faculty: Fetch attendance
app.get("/faculty/attendance", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Access denied" });
  const facultyId = req.user.faculty_id;

  try {
    const result = await db.query(`
      SELECT a.attendance_id, a.timestamp,
             s.student_id, s.name AS student_name,
             sub.subject_id, sub.name AS subject_name
      FROM attendance a
      JOIN students s ON a.student_id = s.student_id
      JOIN subjects sub ON a.subject_id = sub.subject_id
      WHERE sub.faculty_id = $1
      ORDER BY a.timestamp DESC
    `, [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch attendance" });
  }
});

// Add attendance
app.post("/faculty/attendance", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const { student_id, subject_id, subject_name } = req.body;
  if (!student_id || !subject_id || !subject_name)
    return res.status(400).json({ error: "Student ID, Subject ID, and Subject Name are required" });

  try {
    const result = await db.query(`
      INSERT INTO attendance (student_id, subject_id, subject_name)
      VALUES ($1, $2, $3) RETURNING attendance_id
    `, [student_id, subject_id, subject_name]);
    res.json({ message: "Attendance added", attendance_id: result.rows[0].attendance_id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update attendance
app.put("/faculty/attendance/:id", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const { id } = req.params;
  const { student_id, subject_id, subject_name, timestamp } = req.body;

  try {
    const result = await db.query(`
      UPDATE attendance
      SET student_id=$1, subject_id=$2, subject_name=$3, timestamp=$4
      WHERE attendance_id=$5 RETURNING *
    `, [student_id, subject_id, subject_name, timestamp, id]);

    if (result.rowCount === 0) return res.status(404).json({ error: "Attendance not found" });
    res.json({ message: "Attendance updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete attendance
app.delete("/faculty/attendance/:id", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });
  const { id } = req.params;

  try {
    const result = await db.query("DELETE FROM attendance WHERE attendance_id=$1 RETURNING *", [id]);
    if (result.rowCount === 0) return res.status(404).json({ error: "Attendance not found" });
    res.json({ message: "Attendance deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Create user login first (no student_id yet)
// Register user (creates a new student user)
app.post("/register-user", async (req, res) => {
  const { username, password } = req.body;

  try {
    const result = await db.query(
      "INSERT INTO users (username, password, role) VALUES ($1, $2, 'student') RETURNING user_id",
      [username, password]
    );
    res.json({
      message: "User created, please login and complete registration",
      user_id: result.rows[0].user_id
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Student self-registration
// ---------------- Student Self-Registration ----------------
app.post("/students", authenticateStudent, async (req, res) => {
  // Ensure logged-in user is a student
  if (req.user.role !== "student") {
    return res.status(403).json({ error: "Only students can self-register" });
  }

  // Get logged-in username from JWT
  const username = req.user.username;
  if (!username) {
    return res.status(400).json({ error: "Logged-in username missing from token." });
  }

  // Get student info from request body
  const { name, roll_number, class_name, email, phone } = req.body;

  try {
    // Step 1: Insert student record
    const studentResult = await db.query(
      `INSERT INTO students (name, roll_number, class, email, phone, registered_on)
       VALUES ($1, $2, $3, $4, $5, NOW())
       RETURNING student_id`,
      [name, roll_number, class_name, email, phone]
    );
    const newStudentId = studentResult.rows[0].student_id;

    // Step 2: Link this student_id to the user
    const updateResult = await db.query(
      `UPDATE users
       SET student_id = $1
       WHERE username = $2
         AND role = 'student'
         AND student_id IS NULL
       RETURNING user_id`,
      [newStudentId, username]
    );

    if (updateResult.rowCount === 0) {
      return res.status(400).json({
        error: `No matching unlinked user found for '${username}'.`,
      });
    }

    // Step 3: Optional welcome email
    try {
      await sendMail(
        email,
        "🎓 Welcome to Smart Attendance System",
        `Hello ${name},\n\nYou have successfully registered in the Smart Attendance System.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nYou can now log in and start marking your attendance.\n\nBest Regards,\nAdmin Team`,
        `<h2>Welcome, ${name}! 🎓</h2>
         <p>You have successfully registered in the <b>Smart Attendance System</b>.</p>
         <p>
           <b>Roll Number:</b> ${roll_number}<br>
           <b>Class:</b> ${class_name}<br>
           <b>Student ID:</b> ${newStudentId}
         </p>
         <p>You can now log in and start marking your attendance.</p>
         <br>
         <p>Best Regards,<br><b>Admin Team</b></p>`
      );
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    // Step 4: Send success response
    res.json({
      success: true,
      message: "✅ Student registered successfully and linked to user!",
      student_id: newStudentId,
    });
  } catch (err) {
    console.error("❌ Registration error:", err);
    res.status(500).json({ error: err.message });
  }
});



// Alternative API endpoint to register student & link user
app.post("/api/students", async (req, res) => {
  const { name, roll_number, class_name, email, phone, username } = req.body;

  try {
    // Step 1: Insert student
    const studentResult = await db.query(
      `INSERT INTO students (name, roll_number, class, email, phone, registered_on)
       VALUES ($1, $2, $3, $4, $5, NOW()) RETURNING student_id`,
      [name, roll_number, class_name, email, phone]
    );
    const newStudentId = studentResult.rows[0].student_id;

    // Step 2: Update user
    const updateResult = await db.query(
      `UPDATE users
       SET student_id = $1
       WHERE username = $2 AND student_id IS NULL
       RETURNING user_id`,
      [newStudentId, username]
    );

    if (updateResult.rowCount === 0) {
      console.warn(`⚠️ No user found with username '${username}' or student_id already set.`);
    }

    // Step 3: Send email
    try {
      await sendMail({
        to: email,
        subject: "🎉 Registration Successful - Smart Attendance System",
        text: `Hello ${name},\n\nYou have been successfully registered in the Smart Attendance System.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nYou can now log in using your username: ${username}.\n\nBest Regards,\nAdmin Team`,
      });
      console.log(`📧 Registration email sent to ${email}`);
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    res.json({
      success: true,
      message: "✅ Student registered, linked to user, and email sent successfully!",
      student_id: newStudentId,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ✅ Student Timetable Route
// Get timetable for logged-in student
app.get('/student/timetable', authenticateToken, async (req, res) => {
  if (req.user.role !== 'student') return res.status(403).json({ error: "Forbidden" });

  const sql = `
    SELECT t.timetable_id, t.day, t.start_time, t.end_time, s.name AS subject
    FROM student_timetable st
    JOIN timetable t ON st.timetable_id = t.timetable_id
    JOIN subjects s ON t.subject_id = s.subject_id
    WHERE st.student_id = $1
    ORDER BY 
      CASE t.day
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
      END,
      t.start_time
  `;

  try {
    const result = await db.query(sql, [req.user.student_id]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get timetable for logged-in faculty
app.get("/faculty/timetable", authenticateFaculty, async (req, res) => {
  const facultyId = req.user.faculty_id;

  const sql = `
    SELECT t.timetable_id, t.subject_id, s.name AS subject_name, 
           t.day, t.start_time, t.end_time
    FROM timetable t
    JOIN subjects s ON t.subject_id = s.subject_id
    WHERE s.faculty_id = $1
    ORDER BY 
      CASE t.day
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
      END,
      t.start_time
  `;

  try {
    const result = await db.query(sql, [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Add timetable entry (faculty only)
app.post("/faculty/timetable", authenticateFaculty, async (req, res) => {
  const facultyId = req.user.faculty_id;
  const { subject_id, day, start_time, end_time } = req.body;

  try {
    // Validate subject belongs to this faculty
    const check = await db.query(
      "SELECT 1 FROM subjects WHERE subject_id = $1 AND faculty_id = $2",
      [subject_id, facultyId]
    );
    if (check.rowCount === 0) return res.status(403).json({ error: "Not authorized for this subject" });

    // Insert timetable row
    const insertResult = await db.query(
      `INSERT INTO timetable (faculty_id, subject_id, day, start_time, end_time)
       VALUES ($1, $2, $3, $4, $5) RETURNING timetable_id`,
      [facultyId, subject_id, day, start_time, end_time]
    );
    const timetableId = insertResult.rows[0].timetable_id;

    // Insert into student_timetable for assigned students (ignore duplicates)
    await db.query(
      `INSERT INTO student_timetable (student_id, timetable_id)
       SELECT fs.student_id, $1
       FROM faculty_students fs
       WHERE fs.faculty_id = $2
       ON CONFLICT (student_id, timetable_id) DO NOTHING`,
      [timetableId, facultyId]
    );

    res.json({ message: "Timetable entry added successfully for faculty and assigned students" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update timetable entry (faculty only)
app.put("/faculty/timetable/:id", authenticateFaculty, async (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;
  const { day, start_time, end_time } = req.body;

  try {
    const result = await db.query(
      `UPDATE timetable t
       SET day = $1, start_time = $2, end_time = $3
       FROM subjects s
       WHERE t.subject_id = s.subject_id
       AND t.timetable_id = $4
       AND s.faculty_id = $5`,
      [day, start_time, end_time, timetableId, facultyId]
    );

    if (result.rowCount === 0) return res.status(403).json({ error: "Not authorized" });
    res.json({ message: "Timetable updated successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete timetable entry (faculty only)
app.delete("/faculty/timetable/:id", authenticateFaculty, async (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;

  try {
    // Delete related student timetable entries first
    await db.query("DELETE FROM student_timetable WHERE timetable_id = $1", [timetableId]);

    // Then delete timetable entry (only if owned by faculty)
    const deleteResult = await db.query(
      `DELETE FROM timetable t
       USING subjects s
       WHERE t.subject_id = s.subject_id
       AND t.timetable_id = $1
       AND s.faculty_id = $2`,
      [timetableId, facultyId]
    );

    if (deleteResult.rowCount === 0) return res.status(403).json({ error: "Not authorized" });
    res.json({ message: "Timetable deleted successfully (faculty + students)" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ================= ADMIN ROUTES =================

// Get all users
app.get("/admin/users", authenticateToken, isAdmin, async (req, res) => {
  try {
    const result = await db.query("SELECT user_id, username, role FROM users");
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Add new user
app.post("/admin/addUser", authenticateToken, isAdmin, async (req, res) => {
  const { username, password, role } = req.body;
  if (!username || !password || !role) return res.status(400).json({ error: "All fields required" });

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    await db.query(
      "INSERT INTO users (username, password, role) VALUES ($1, $2, $3)",
      [username, hashedPassword, role]
    );
    res.json({ message: "✅ User added successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete user
app.delete("/admin/users/:id", authenticateToken, isAdmin, async (req, res) => {
  try {
    await db.query("DELETE FROM users WHERE user_id = $1", [req.params.id]);
    res.json({ message: "🗑 User deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- STUDENTS ----------------
app.get("/admin/students", authenticateToken, async (req, res) => {
  try {
    const result = await db.query("SELECT * FROM students");
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/students", authenticateToken, async (req, res) => {
  const { name, roll_number, class: studentClass, email, phone } = req.body;

  try {
    const result = await db.query(
      "INSERT INTO students (name, roll_number, class, email, phone) VALUES ($1,$2,$3,$4,$5) RETURNING student_id",
      [name, roll_number, studentClass, email, phone]
    );
    const newStudentId = result.rows[0].student_id;

    // ✅ Send welcome email
    try {
      await sendMail(
        email,
        "🎉 Registration Successful - Smart Attendance System",
        `Hello ${name},\n\nYou have been successfully registered.\nStudent ID: ${newStudentId}`,
        `<h2>Welcome, ${name}! 🎉</h2>
         <p>Your Student ID: ${newStudentId}</p>`
      );
      console.log(`📧 Registration email sent to ${email}`);
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    res.json({ message: "Student added & email sent", student_id: newStudentId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/admin/students/:id", authenticateToken, async (req, res) => {
  try {
    await db.query("DELETE FROM students WHERE student_id = $1", [req.params.id]);
    res.json({ message: "Student deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- FACULTY ----------------
app.get("/admin/faculty", authenticateToken, async (req, res) => {
  const sql = `
    SELECT f.faculty_id, f.name, f.department, u.user_id
    FROM faculty f
    LEFT JOIN users u ON u.faculty_id = f.faculty_id
  `;
  try {
    const result = await db.query(sql);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/faculty", authenticateToken, async (req, res) => {
  const { user_id, name, department } = req.body;
  if (!user_id) return res.status(400).json({ error: "user_id is required" });

  try {
    const result = await db.query(
      "INSERT INTO faculty (name, department) VALUES ($1, $2) RETURNING faculty_id",
      [name, department]
    );
    const facultyId = result.rows[0].faculty_id;

    await db.query("UPDATE users SET faculty_id = $1 WHERE user_id = $2", [facultyId, user_id]);

    res.json({ message: "Faculty added and linked to user", faculty_id: facultyId, user_id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/admin/faculty/:id", authenticateToken, async (req, res) => {
  try {
    await db.query("DELETE FROM faculty WHERE faculty_id = $1", [req.params.id]);
    res.json({ message: "Faculty deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- SUBJECTS CRUD ---
// Get all subjects
app.get("/admin/subjects", authenticateToken, async (req, res) => {
  try {
    const result = await db.query("SELECT * FROM subjects");
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Unassign subject from faculty
// Must come BEFORE /assign
app.put("/admin/subjects/:id/unassign", authenticateToken, async (req, res) => {
  const subjectId = req.params.id;
  try {
    const result = await db.query(
      "UPDATE subjects SET faculty_id = NULL WHERE subject_id = $1 RETURNING *",
      [subjectId]
    );
    res.json({ message: "Subject unassigned successfully", result: result.rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put("/admin/subjects/:id/assign", authenticateToken, async (req, res) => {
  const subjectId = req.params.id;
  const { faculty_id } = req.body;

  try {
    const result = await db.query(
      "UPDATE subjects SET faculty_id = $1 WHERE subject_id = $2 RETURNING *",
      [faculty_id, subjectId]
    );
    res.json({ message: "Subject assigned successfully", result: result.rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/subjects", authenticateToken, async (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: "Subject name is required" });

  try {
    const result = await db.query(
      "INSERT INTO subjects (name) VALUES ($1) RETURNING subject_id, name",
      [name]
    );
    res.json({
      message: "Subject added successfully",
      subject: result.rows[0]
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update subject (only name)
app.put("/admin/subjects/:id", authenticateToken, async (req, res) => {
  const subjectId = req.params.id;
  const { name } = req.body;

  if (!name) return res.status(400).json({ error: "New subject name is required" });

  try {
    const result = await db.query(
      "UPDATE subjects SET name = $1 WHERE subject_id = $2 RETURNING *",
      [name, subjectId]
    );
    res.json({ message: "Subject updated successfully", result: result.rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/admin/subjects/:id", authenticateToken, async (req, res) => {
  const { id } = req.params;
  try {
    await db.query("DELETE FROM subjects WHERE subject_id = $1", [id]);
    res.json({ message: "Subject deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- DELETE FACULTY ----------------
app.delete("/admin/faculty/:id", authenticateToken, async (req, res) => {
  const facultyId = req.params.id;

  try {
    // Unassign subjects
    await db.query("UPDATE subjects SET faculty_id = NULL WHERE faculty_id = $1", [facultyId]);

    // Delete faculty
    await db.query("DELETE FROM faculty WHERE faculty_id = $1", [facultyId]);
    res.json({ message: "Faculty deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- ASSIGN STUDENTS ----------------
app.get("/admin/faculty/:facultyId/assigned-students", authenticateToken, async (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });

  const { facultyId } = req.params;

  const sql = `
    SELECT fs.id, s.student_id, s.name, s.roll_number, s.class, s.email, s.phone, fs.assigned_on
    FROM faculty_students fs
    JOIN students s ON fs.student_id = s.student_id
    WHERE fs.faculty_id = $1
  `;

  try {
    const result = await db.query(sql, [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});



app.post("/admin/faculty/:facultyId/assigned-students", authenticateToken, async (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });
  const { facultyId } = req.params;
  const { student_id } = req.body;

  try {
    // Insert with ON CONFLICT DO NOTHING
    await db.query(
      "INSERT INTO faculty_students (faculty_id, student_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
      [facultyId, student_id]
    );

    // Add timetable entries for student
    await db.query(
      `INSERT INTO student_timetable (student_id, timetable_id)
       SELECT $1, t.timetable_id
       FROM timetable t
       JOIN subjects sub ON t.subject_id = sub.subject_id
       WHERE sub.faculty_id = $2
       ON CONFLICT DO NOTHING`,
      [student_id, facultyId]
    );

    res.json({ message: "✅ Student assigned and timetable updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/admin/faculty/:facultyId/assigned-students/:studentId", authenticateToken, async (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });
  const { facultyId, studentId } = req.params;

  try {
    const result = await db.query(
      "DELETE FROM faculty_students WHERE faculty_id = $1 AND student_id = $2",
      [facultyId, studentId]
    );

    if (result.rowCount === 0) return res.status(404).json({ error: "Not assigned" });

    await db.query(
      `DELETE FROM student_timetable st
       USING timetable t, subjects sub
       WHERE st.timetable_id = t.timetable_id
       AND t.subject_id = sub.subject_id
       AND st.student_id = $1 AND sub.faculty_id = $2`,
      [studentId, facultyId]
    );

    res.json({ message: "✅ Student unassigned and timetable updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- TIMETABLE ----------------
// ---------------- GET TIMETABLE FOR FACULTY ----------------
app.get("/admin/timetable/faculty/:facultyId", authenticateToken, async (req, res) => {
  const { facultyId } = req.params;
  const sql = `
    SELECT 
      t.timetable_id, 
      s.subject_id, 
      s.name AS subject, 
      t.day, 
      t.start_time, 
      t.end_time
    FROM timetable t
    JOIN subjects s ON t.subject_id = s.subject_id
    WHERE s.faculty_id = $1
  `;

  try {
    const result = await db.query(sql, [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- ADD TIMETABLE ----------------
app.post("/admin/timetable", authenticateToken, async (req, res) => {
  const { faculty_id, subject_id, day, start_time, end_time } = req.body;

  try {
    const insertSql = `
      INSERT INTO timetable (faculty_id, subject_id, day, start_time, end_time)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING timetable_id
    `;
    const result = await db.query(insertSql, [faculty_id, subject_id, day, start_time, end_time]);
    const timetableId = result.rows[0].timetable_id;

    // Insert timetable for assigned students
    const studentSql = `
      INSERT INTO student_timetable (student_id, timetable_id)
      SELECT fs.student_id, $1
      FROM faculty_students fs
      WHERE fs.faculty_id = $2
      ON CONFLICT DO NOTHING
    `;
    await db.query(studentSql, [timetableId, faculty_id]);

    res.json({ message: "✅ Timetable added (faculty + students)", timetable_id: timetableId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- UPDATE TIMETABLE ----------------
app.put("/admin/timetable/:id", authenticateToken, async (req, res) => {
  const { id } = req.params;
  const { subject_id, day, start_time, end_time } = req.body;

  try {
    const result = await db.query(
      `UPDATE timetable
       SET subject_id = $1, day = $2, start_time = $3, end_time = $4
       WHERE timetable_id = $5`,
      [subject_id, day, start_time, end_time, id]
    );

    if (result.rowCount === 0)
      return res.status(404).json({ error: "Timetable not found" });

    res.json({ message: "Timetable updated (faculty + students)" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- DELETE TIMETABLE ----------------
app.delete("/admin/timetable/:id", authenticateToken, async (req, res) => {
  const { id } = req.params;

  try {
    await db.query("DELETE FROM student_timetable WHERE timetable_id = $1", [id]);
    const result = await db.query("DELETE FROM timetable WHERE timetable_id = $1", [id]);

    if (result.rowCount === 0)
      return res.status(404).json({ error: "Timetable not found" });

    res.json({ message: "Timetable deleted (faculty + students)" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------- GET SUBJECTS BY FACULTY ----------------
app.get("/admin/subjects/faculty/:facultyId", authenticateToken, async (req, res) => {
  const { facultyId } = req.params;
  try {
    const result = await db.query("SELECT * FROM subjects WHERE faculty_id = $1", [facultyId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ---------------- FULL PIPELINE ----------------
app.get("/pipeline/:name", (req, res) => {
  const name = req.params.name;
  const scripts = [
    "1_capture_images.py",
    "2_crop_faces.py",
    "3_generate_embeddings.py",
    "insert_embedding.py",
  ];

  let outputLog = "";

  const runNext = (i) => {
    if (i >= scripts.length) return res.json({ message: "✅ Pipeline completed", log: outputLog });

    const scriptPath = path.join(__dirname, "python_scripts", scripts[i]);
    const command = `python "${scriptPath}" "${name}"`;

    exec(command, (err, stdout, stderr) => {
      if (err) {
        console.error(`❌ Error running ${scripts[i]}:`, stderr);
        return res.status(500).json({
          error: `Failed at ${scripts[i]}`,
          details: stderr.toString(),
        });
      }

      console.log(`✅ Ran ${scripts[i]} for ${name}`);
      outputLog += `\n----- ${scripts[i]} -----\n${stdout || stderr}\n`;

      runNext(i + 1);
    });
  };

  runNext(0);
});

// ---------------- FACE RECOGNITION ----------------
app.get("/face-recognition", (req, res) => {
  const scriptPath = path.join(__dirname, "python_scripts", "4_face_recognition.py");
  const command = `python "${scriptPath}"`;

  exec(command, (err, stdout, stderr) => {
    if (err) {
      console.error(`❌ Error running face recognition:`, stderr);
      return res.status(500).json({ error: stderr.toString() });
    }

    console.log("✅ Face recognition executed");
    res.json({ output: stdout || stderr });
  });
});

// ---------------- CATCH ALL ----------------
app.use((req, res) => res.status(404).json({ error: "Route not found" }));

// ---------------- START SERVER ----------------
app.listen(port, () => {
  console.log(`🚀 Server running on http://localhost:${port}`);
});
