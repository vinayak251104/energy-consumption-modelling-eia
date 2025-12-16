const fs = require('fs');
const superagent = require('superagent');

// function to write the dataset from api into a csv file
writeFilePro = (file, data) => {
    return new Promise((resolve, reject) => {
        fs.writeFile(file, data, err => {
            if (err) reject('Could not write file 😢');
            else resolve('success');
        });
    });
};

async function extractOperations() {
    try {
        // extract the data from the api itself 
        const API_KEY = process.env.EIA_API_KEY

        if (!API_KEY) {
            throw new Error('Set EIA_API_KEY as an environment variable');
        }

        const res = await superagent.get(
            `https://api.eia.gov/v2/electricity/electric-power-operational-data/data/?api_key=${API_KEY}&frequency=annual&data[7]=generation&data[14]=total-consumption-btu&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000`
        );

        console.log('success');

        const data = res.body;
        // to map the features within the dataset 
        const result = data.response.data.map(el => ({
            period: el.period,
            location: el.location, 
            state: el.stateDescription,
            sector: el.sectorDescription,
            fuel: el.fuelTypeDescription,
            generation: Number(el.generation),
            totalConsumptionBtu: Number(el["total-consumption-btu"])
        }));
        
        await writeFilePro(
            'electricity-data.csv',
            "period,location,state,sector,fuel,generation,totalConsumptionBtu\n" +
            result.map(el =>
                `${el.period},${el.location},${el.state},${el.sector},${el.fuel},${el.generation},${el.totalConsumptionBtu}`
            ).join('\n')
        );

    } catch (err) {
        console.log(err.message);
    }
}

extractOperations();
