import { getTrafficFromTDCam } from './traffic_td_cam.js';
import { getTrafficOfficialAbderdeenTunnel } from './traffic_td_official.js';

getTrafficOfficialAbderdeenTunnel().then(r => console.log(r));

export {
    getTrafficFromTDCam
}