// @ts-ignore
process.env.OPENCV4NODEJS_DISABLE_EXTERNAL_MEM_TRACKING = 1;

const fs = require('fs');
const cv = require('opencv4nodejs');
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')();

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const FPS = 10;
const TIMECOMPARE = 3;
const PORT = 3000;

let comparators = [];
let trainersArr = [];
let recognizer;

/**
 * @param {string} path
 * @param {any[]} arr
 */
async function readfileAsync(path, arr) {
	return cv.imreadAsync(path).then(res => onReadFile(res, path, arr));
}

/**
 * @param {any} img
 * @param {string} path
 * @param {{ grey: any; name: any; }[]} arr
 */
function onReadFile(img, path, arr) {
	return img.bgrToGrayAsync().then(grey => onImage(grey, path, arr));
}

/**
 * @param {any} imgGrey
 * @param {string} path
 * @param {{ grey: any; name: any; }[]} arr
 */
function onImage(imgGrey, path, arr) {
	const paths = path.split('/');
	const folderPerson = paths[paths.length - 2];
	arr.push({ grey: imgGrey, name: folderPerson });
	console.log(folderPerson, 'processed');
}

/**
 * @param {{ label: any; confidence: any; }} res
 * @param {{grey: any; name: number}[]} trainers
 */
function onPrediction(res, trainers) {
	const name = trainers.filter((x, i) => i == res.label).map(v => v.name)[0];
	const prediction = { ...res, name };

	if (comparators.length == FPS * TIMECOMPARE) {
		// @ts-ignore
		const min = comparators.reduce(minExpression);
		// console.log(min);
		console.log(`hi ${min.name} confidence: ${min.confidence}`);
		comparators = [];

		io.emit('hello_man', min.name);
	}

	if (res.confidence < 100) {
		comparators.push(prediction);
	}
}

let startTime, endTime;

function start() {
	startTime = new Date();
}

function end() {
	endTime = new Date();
	// @ts-ignore
	let timeDiff = endTime - startTime; //in ms
	// strip the ms
	timeDiff /= 1000;

	// get seconds
	const seconds = Math.round(timeDiff);
	return seconds;
}

async function onEncodedAsync(img) {
	const base64 = img.toString('base64');
	io.emit('image', base64);
}

/**
 * @param {Buffer}buffer
 */
async function onFrameAsync(buffer) {
	// receve buffer da analizzare
	const frame = await cv.imdecodeAsync(buffer);
	// non è necessario perché arriva già
	// un immagine in formato grey
	let grey = await frame.bgrToGrayAsync();
	const { objects } = await classifier.detectMultiScaleAsync(grey);

	let rect;
	let rectFounded = false;

	if (!objects || objects.length == 0) {
		// console.error(`${path} non ho riconosciuto nessun viso`);
		rect = null;
	} else if (objects.length > 1) {
		// console.log(`ho trovate più di un viso`);
		rect = null;
	} else {
		// viso corretto
		rect = objects[0];
	}

	if (rect) {
		grey = grey.getRegion(rect);
		frame.drawRectangle(rect, new cv.Vec3(0, 255, 0));
		rectFounded = true;
	}

	cv.imencodeAsync('.jpg', frame)
		.then(res => onEncodedAsync(res))
		.catch(err => console.log(err));

	if (!rectFounded) {
		return;
	}

	recognizer
		.predictAsync(grey)
		.then(res => onPrediction(res, trainersArr))
		.catch(err => console.error(err));
}

async function initServerAsync() {
	console.log('[SERVER] server starting...');
	app.get('/', (req, res) => {
		res.sendFile(path.join(__dirname, 'index.html'));
	});

	app.listen(PORT + 1);
	io.attach(PORT);
	io.on('connection', socket => {
		socket.on('imageToAnalyze', data => onFrameAsync(data));
	});
	// TODO: Non funziona! capire il perchè
	// io.on('ping', () => console.log('ping'));
	// io.on('reconnecting', () => console.log('reconnecting'));
	// io.on('error', error => console.log('error', error));
	// io.on('connect_error', err => console.log('connect_error', err));
	// // server.listen(PORT);
	// // console.log(`[SERVER] started and listening on ${PORT} port`);

	// // io.on('image', data => onFrameAsync(data));
	// io.on('imageToAnalyze', () => console.log('data received'));
	// io.on('server-connected', res => console.log(res));
	// // console.log('[SOCKET] websocket started');
	// io.emit('socket-started', true);
}

async function initRecognitionAsync() {
	// using async await
	try {
		start();
		console.log(`[PROCESS] start processing images...`);

		const realtivepath = path.join(__dirname, '/images/trainers');
		const trainers = fs.readdirSync(realtivepath);
		let trainersPromiseArr = [];

		// remove DS_STORE if exist
		if (trainers[0] == '.DS_Store') {
			trainers.splice(0, 1);
		}

		for (let i = 0; i < trainers.length; i++) {
			const imaegsFolder = path.join(realtivepath, trainers[i]);
			const images = fs.readdirSync(imaegsFolder);

			// remove DS_STORE if exist
			if (images[0] == '.DS_Store') {
				images.splice(0, 1);
			}

			for (let j = 0; j < images.length; j++) {
				trainersPromiseArr.push(
					readfileAsync(
						path.join(imaegsFolder, images[j]),
						trainersArr
					)
				);
			}
		}

		await Promise.all(trainersPromiseArr);

		recognizer = new cv.LBPHFaceRecognizer();

		await recognizer.trainAsync(
			trainersArr.map(v => v.grey),
			trainersArr.map((v, i) => i)
		);

		console.log(`[PROCESS] initialized at ${end()} second`);
	} catch (err) {
		console.error(err);
	}
}

async function initAsync() {
	// initalize server
	await initServerAsync();
	// initalize recognition opencv
	await initRecognitionAsync();
}

function minExpression(min, next) {
	return min.confidence < next.confidence ? min : next;
}

initAsync();
