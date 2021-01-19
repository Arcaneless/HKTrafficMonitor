var { spawn } = require('child_process');

function getTrafficFromTDCam() {
    return new Promise((resolve, reject) => {
        try {
            const pythonProcess = spawn('python', ['traffic.py'])
            pythonProcess.stdout.on('data', (data) => {
                resolve(JSON.parse(data.toString().replace(/\'/g, '\"')));
            });
        } catch (e) {
            console.warn(e);
            reject(e);
        }
    });
}

module.exports = {
    getTrafficFromTDCam
};