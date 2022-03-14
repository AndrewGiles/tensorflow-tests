const LEFT = 'LEFT';
const RIGHT = 'RIGHT';
const UP = 'UP';
const DOWN = 'DOWN';

const ACTION_NAMES = {
    LEFT,
    RIGHT,
    UP,
    DOWN,
};

const ACTIONS = {
    LEFT: 0,
    RIGHT: 1,
    UP: 2,
    DOWN: 3,
}

const ACTIONS_LENGTH = Object.values(ACTIONS).length;

const MAX_STEPS = 50;

const MAX_EPS = 1;
const MIN_EPS = 0.1;

let eps = MAX_EPS;

const model = createOrRetrieveModel();
tfvis.show.modelSummary({ name: 'Model Summary' }, model);

let trainingHistory = [];

async function run() {

    let gameNum = 0

    while (gameNum < 1000) {

        const gameLength = await trainGame();

        trainingHistory.push(gameLength);

        const values = trainingHistory.map((matchLength, i) => ({
            x: i,
            y: matchLength,
        }));

        tfvis.render.scatterplot(
            { name: 'Game # v Game Length' },
            { values },
            {
                xLabel: 'Game #',
                yLabel: 'Length',
                height: 300
            }
        );
        gameNum++;
    }



    // // More code will be added below
    // // Convert the data to a form we can use for training.
    // const tensorData = convertToTensor(data);
    // const { inputs, labels } = tensorData;

    // // Train the model
    // await trainModel(model, inputs, labels);
    console.log('Done Training');

    // // Make some predictions using the model and compare them to the
    // // original data
    // testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);

async function trainGame() {
    let gameHistory = [];

    let gameBoard = createNewMinimalBoard();

    // We don't want to start with a "winning" board.
    while (hasWon(gameBoard)) {
        gameBoard = createNewMinimalBoard();
    }

    let step = 0;
    // once we have a valid starting state, we want to train the model
    while (step < MAX_STEPS && !hasWon(gameBoard)) {
        const availableActions = getAvailableActions(gameBoard);
        const actionIndex = await predictAction(model, gameBoard, availableActions);
        const reward = hasWon(gameBoard) ? 10 : -1;
        const newBoard = takeAction(gameBoard, actionIndex);
        gameHistory.push([gameBoard, actionIndex, reward, newBoard]);
        gameBoard = newBoard;
        step++;
    }
    eps = Math.max(eps - 0.01, MIN_EPS);
    
    await trainModel(gameHistory);

    return gameHistory.length;
}

function createNewMinimalBoard() {
    const selectedPos = {
        x: Math.floor(Math.random() * 8),
        y: Math.floor(Math.random() * 4),
    }
    const board = createNewBoard(selectedPos);

    return board;
}

function createNewBoard(piecePosition) {
    return Array(4).fill().map((row, y) => (
        Array(8).fill().map((col, x) => (
            ((x === piecePosition.x) && (y === piecePosition.y)) ? 1 : 0
        ))
    ));
}

function hasWon(board) {
    if ((board[0][0] === 1) || (board[3][7] === 1)) return true;
    return false;
}

function getAvailableActions(board) {
    const currentLocation = getCurrentLocation(board);
    const availableActions = [];
    if (currentLocation.x > 0) availableActions.push(ACTIONS.LEFT);
    if (currentLocation.x < 7) availableActions.push(ACTIONS.RIGHT);
    if (currentLocation.y > 0) availableActions.push(ACTIONS.DOWN);
    if (currentLocation.y < 3) availableActions.push(ACTIONS.UP);
    return availableActions;
}

async function predictAction(model, gameBoard, availableActions) {
    if (Math.random() < eps) {
        const availableActionIndex = Math.floor(Math.random() * availableActions.length);
        return availableActions[availableActionIndex];
    }

    const predictions = await model.predict(convertBoardToTensor(gameBoard));
    const validPredictions = predictions.dataSync().map((prediction, i) => availableActions.includes(i) ? prediction : -1);
    console.log({validPredictions})
    return tf.tensor(validPredictions).argMax().dataSync()[0];
}

function takeAction(board, actionIndex) {
    const currentLocation = getCurrentLocation(board);
    const newLocation = getNewLocation(currentLocation, actionIndex);
    return createNewBoard(newLocation);
}

function getCurrentLocation(board) {
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
            if (board[y][x] === 1) {
                return { x, y };
            }
        }
    }
}

function getNewLocation(currentLocation, actionIndex) {
    const newLocation = {
        x: currentLocation.x,
        y: currentLocation.y,
    }
    switch (actionIndex) {
        case ACTIONS.LEFT:
            newLocation.x--;
            break;
        case ACTIONS.RIGHT:
            newLocation.x++;
            break;
        case ACTIONS.DOWN:
            newLocation.y--;
            break;
        case ACTIONS.UP:
            newLocation.y++;
            break;
    }
    return newLocation;
}

function createOrRetrieveModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ name: 'input-layer', inputShape: [4, 8], units: 25 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ name: 'hidden-1', units: 75, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ name: 'hidden-2', units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ name: 'hidden-3', units: 25, activation: 'sigmoid' }));

    // Add an output layer
    model.add(tf.layers.dense({ name: 'output-layer', units: 4, activation: 'softmax' }));

    return model;
}

function convertBoardToTensor(board) {
    return tf.tensor3d([board], [1,4, 8]);
}

function convertToTensor(gameHistory) {
    // Step 1. Shuffle the data
    tf.util.shuffle(gameHistory);

    // Step 2. Convert data to Tensor
    const inputs = gameHistory.map(([state]) => state)
    const labels = gameHistory.map(([, actionIndex,]) => tf.oneHot(actionIndex, ACTIONS_LENGTH).dataSync());
    const rewards = gameHistory.map(([, , rewards]) => rewards);

    const inputTensor = tf.tensor3d(inputs, [inputs.length, inputs[0].length, inputs[0][0].length]);
    const labelTensor = tf.tensor2d(labels, [labels.length, labels[0].length]);
    const rewardsTensor = tf.tensor2d(rewards, [rewards.length, 1]);

    return {
        inputs: inputTensor,
        labels: labelTensor,
        rewards: rewardsTensor,
    }
}

async function trainModel(gameHistory) {
    // Prepare the model for training.
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
    });

    const batchSize = 32;
    const epochs = 5;

    const { inputs, labels, rewards } = convertToTensor(gameHistory);

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        weights: rewards,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

// function testModel(model, inputData, normalizationData) {
//     const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

//     // Generate predictions for a uniform range of numbers between 0 and 1;
//     // We un-normalize the data by doing the inverse of the min-max scaling
//     // that we did earlier.
//     const [xs, preds] = tf.tidy(() => {

//         const xs = tf.linspace(0, 1, 100);
//         const preds = model.predict(xs.reshape([100, 1]));

//         const unNormXs = xs
//             .mul(inputMax.sub(inputMin))
//             .add(inputMin);

//         const unNormPreds = preds
//             .mul(labelMax.sub(labelMin))
//             .add(labelMin);

//         // Un-normalize the data
//         return [unNormXs.dataSync(), unNormPreds.dataSync()];
//     });


//     const predictedPoints = Array.from(xs).map((val, i) => {
//         return { x: val, y: preds[i] }
//     });

//     const originalPoints = inputData.map(d => ({
//         x: d.horsepower, y: d.mpg,
//     }));


//     tfvis.render.scatterplot(
//         { name: 'Model Predictions vs Original Data' },
//         { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
//         {
//             xLabel: 'Horsepower',
//             yLabel: 'MPG',
//             height: 300
//         }
//     );
// }

