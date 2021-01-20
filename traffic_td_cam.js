import { spawn } from 'child_process';

export function getTrafficFromTDCam() {
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