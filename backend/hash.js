const bcrypt = require("bcryptjs");

(async () => {
  const studentPass = await bcrypt.hash("student123", 10);   // password for nehan
  const facultyPass = await bcrypt.hash("faculty123", 10);   // password for professor1

  console.log("Student hash:", studentPass);
  console.log("Faculty hash:", facultyPass);
})();
