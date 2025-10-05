function runScript(scriptName) {
  fetch(`/run/${scriptName}`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      document.getElementById("response").textContent = data.output;
    })
    .catch((error) => {
      document.getElementById("response").textContent = `❌ Error:\n${error}`;
    });
}

const token = localStorage.getItem("token"); // JWT from login

// Fetch students table
async function fetchStudents() {
    const res = await fetch("http://localhost:3000/faculty/students", {
        headers: { "Authorization": "Bearer " + token }
    });
    const students = await res.json();

    const tbody = document.querySelector("#studentsTable tbody");
    tbody.innerHTML = "";
    students.forEach(s => {
        const row = `<tr>
            <td>${s.student_id}</td>
            <td>${s.name}</td>
            <td>${s.roll_number}</td>
            <td>${s.class}</td>
            <td>${s.email}</td>
            <td>${s.phone}</td>
            <td>
                <button onclick="deleteStudent(${s.student_id})">🗑 Delete</button>
            </td>
        </tr>`;
        tbody.innerHTML += row;
    });
}

// Fetch attendance table
async function fetchAttendance() {
    const res = await fetch("http://localhost:3000/faculty/attendance", {
        headers: { "Authorization": "Bearer " + token }
    });
    const records = await res.json();

    const tbody = document.querySelector("#attendanceTable tbody");
    tbody.innerHTML = "";
    records.forEach(r => {
        const row = `<tr>
            <td>${r.attendance_id}</td>
            <td>${r.student_id}</td>
            <td>${r.name}</td>
            <td>${r.timestamp}</td>
            <td>
                <button onclick="deleteAttendance(${r.attendance_id})">🗑 Delete</button>
            </td>
        </tr>`;
        tbody.innerHTML += row;
    });
}

// Delete a student
async function deleteStudent(id) {
    if (!confirm("Are you sure you want to delete this student?")) return;

    await fetch(`http://localhost:3000/faculty/students/${id}`, {
        method: "DELETE",
        headers: { "Authorization": "Bearer " + token }
    });
    fetchStudents();
}

// Delete an attendance record
async function deleteAttendance(id) {
    if (!confirm("Are you sure you want to delete this attendance record?")) return;

    await fetch(`http://localhost:3000/faculty/attendance/${id}`, {
        method: "DELETE",
        headers: { "Authorization": "Bearer " + token }
    });
    fetchAttendance();
}

// Add new student
async function addStudent() {
    const name = prompt("Enter name:");
    const roll = prompt("Enter roll number:");
    const className = prompt("Enter class:");
    const email = prompt("Enter email:");
    const phone = prompt("Enter phone:");

    if (!name || !roll) return alert("Name and Roll are required!");

    await fetch("http://localhost:3000/faculty/students", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token
        },
        body: JSON.stringify({ name, roll_number: roll, class: className, email, phone })
    });

    fetchStudents();
}

async function runFaceAttendance() {
    const log = document.getElementById("log");
    log.innerText = "⏳ Running face recognition + timetable check...\n";

    try {
        const res = await fetch("http://localhost:3000/face-attendance", {
            headers: { "Authorization": "Bearer " + token }
        });
        const data = await res.json();

        log.innerText += data.log || data.message;
        fetchMyAttendance(); // refresh table
    } catch (err) {
        log.innerText += "\n❌ Request failed: " + err;
    }
}


// Logout
function logout() {
    localStorage.removeItem("token");
    window.location.href = "pipeline.html";
}

// Load data when page loads
fetchStudents();
fetchAttendance();
