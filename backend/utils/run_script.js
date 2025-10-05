const { exec } = require('child_process');
const path = require('path');

function runPythonScript(scriptName, res) {
    const scriptPath = path.join(__dirname, '../../python-scripts', scriptName);  // Go up 2 levels

    exec(`python3 "${scriptPath}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`❌ Error: ${error.message}`);
            return res.status(500).send(`Error executing script: ${error.message}`);
        }
        if (stderr) {
            console.error(`⚠️ Script stderr: ${stderr}`);
        }

        console.log(`✅ Script Output:\n${stdout}`);
        res.send({ output: stdout });
    });
}

module.exports = runPythonScript;
