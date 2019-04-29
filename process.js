const cv = require('opencv4nodejs');
const path = require('path');
const fs = require('fs');

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const totrainerpath = path.join(__dirname, '/images/to_trainers');
const trainerpath = path.join(__dirname, '/images/trainers');

async function initialize() {
	const totrainers = fs.readdirSync(totrainerpath);
	const trainersPromiseArr = [];

	// remove DS_STORE if exist
	if (totrainers[0] == '.DS_Store') {
		totrainers.splice(0, 1);
	}

	for (let i = 0; i < totrainers.length; i++) {
		const imaegsFolder = path.join(totrainerpath, totrainers[i]);
		const images = fs.readdirSync(imaegsFolder);

		// remove DS_STORE if exist
		if (images[0] == '.DS_Store') {
			images.splice(0, 1);
		}

		for (let j = 0; j < images.length; j++) {
			// trainersPromiseArr.push(
			// 	detectFaceSync(path.join(imaegsFolder, images[j]))
			// );
			await detectFaceSync(path.join(imaegsFolder, images[j]));
		}
	}

	// chiamando in async va in out of memory
	// await Promise.all(trainersPromiseArr);
}

async function detectFaceSync(path) {
	try {
		const newpath = path.replace(totrainerpath, trainerpath);
		const folder = newpath.substring(0, newpath.lastIndexOf('/'));

		if (fs.existsSync(newpath)) {
			// return console.log(`${newpath} già processato`);
			return;
		}

		const img = await cv.imreadAsync(path);
		const grey = await img.bgrToGrayAsync();
		const { objects } = await classifier.detectMultiScaleAsync(grey);

		if (!objects || objects.length == 0) {
			console.error(`${path} non ho riconosciuto nessun viso`);
			return;
		}

		if (objects.length > 1) {
			// console.log(`${path} ho trovate più di un viso`);
			// return;
		}

		const imgCropped = grey.getRegion(objects[0]);

		if (!fs.existsSync(folder)) {
			fs.mkdirSync(folder);
		}

		await cv.imwriteAsync(newpath, imgCropped);
		console.log(`${newpath} processato`);
	} catch (err) {
		console.error(err);
	}
}

initialize();
