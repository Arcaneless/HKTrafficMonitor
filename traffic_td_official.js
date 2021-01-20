import axios from 'axios';
import convert from 'xml-js';

// const linkAberdeenTunnelToWanchai = "4647-46478";
const linkAberdeenTunnel = "46478-46479";
const descAberdeenTunnel = "香港島 灣仔區 香港仔隧道灣仔入口"; 
const speedMapURL = "https://resource.data.one.gov.hk/td/speedmap.xml";
const saturationMap = {
    "TRAFFIC GOOD": 1,
    "TRAFFIC AVERAGE": 2,
    "TRAFFIC BAD": 3
};

async function getTrafficOfficial(id) {
    try {
        const { data } = await axios(speedMapURL);
        const xml = JSON.parse(convert.xml2json(data, { compact: true }));
        const result = xml["jtis_speedlist"]["jtis_speedmap"].filter((item) => {
            return item["LINK_ID"]["_text"] === id;
        });

        if (result.length > 0) {
            return {
                saturation: saturationMap[result[0]["ROAD_SATURATION_LEVEL"]["_text"]],
                speed: result[0]["TRAFFIC_SPEED"]["_text"]
            };
        } else {
            return null;
        }
    } catch (e) {
       console.warn(e); 
    }
}

export async function getTrafficOfficialAbderdeenTunnel() {
    const result = await getTrafficOfficial(linkAberdeenTunnel);
    return {
        desc: descAberdeenTunnel,
        state: result.saturation,
        speed: result.speed
    }
}
