const LEFT = 'LEFT';
const RIGHT = 'RIGHT';
const UP = 'UP';
const DOWN = 'DOWN';

const MODEL_LOCATION = 'localstorage://my-model-1'

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
const MIN_EPS = 0.05;
const EP_LOSS = 0.001;
const DISCOUNT_RATE = 0.95;
const TRAINING_CYCLES = 1000;
const LEARNING_RATE = 0.002;

let eps = MAX_EPS;

const IS_TRAINING = true;

const app = new Vue({
    el: '#app',
    data: {
        trainingHistory: [],
        gameHistory: [],
        eps,
        trainingNumber: 0,
    },
    mounted() {
        this.run();
    },
    methods: {
        async run() {
            const model = await createOrRetrieveModel();
            tfvis.show.modelSummary({ name: 'Model Summary' }, model);

            if (IS_TRAINING) {
                while (this.trainingNumber < TRAINING_CYCLES) {

                    const game = await this.trainGame(model);

                    this.trainingHistory = [...this.trainingHistory, game.length].slice(-50);

                    this.trainingNumber++;

                    // if (this.trainingNumber % 100 === 0) {
                    //     await model.save(MODEL_LOCATION);
                    //     console.log('SAVED MODEL');
                    // }
                    if (this.trainingNumber % 100 === 0) this.gameHistory = game;
                }
            } else {
                this.eps = 0;
                const game = await this.trainGame(model);
                console.log({ game: game.map(([board]) => board) })
                this.gameHistory = game;
            }
        },
        async trainGame(model) {
            tf.engine().startScope();
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
                const actionIndex = await predictAction(model, gameBoard, this.eps, availableActions);
                const reward = hasWon(gameBoard) ? 50 : -1;
                const newBoard = takeAction(gameBoard, actionIndex);
                gameHistory.push([gameBoard, actionIndex, reward, newBoard]);
                gameBoard = newBoard;
                step++;
            }
            this.eps = Math.max(this.eps - EP_LOSS, MIN_EPS);

            await trainModel(model, [...gameHistory]);

            tf.engine().endScope();

            return gameHistory;
        }
    }
})

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

async function predictAction(model, gameBoard, eps, availableActions) {
    if (Math.random() < eps) {
        const availableActionIndex = Math.floor(Math.random() * availableActions.length);
        return availableActions[availableActionIndex];
    }

    const predictions = await model.predict(convertBoardToTensor(gameBoard));
    const validPredictions = predictions.dataSync().map((prediction, i) => availableActions.includes(i) ? prediction : -1);
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

async function createOrRetrieveModel() {

    try {
        const model = await tf.loadLayersModel(MODEL_LOCATION);
        console.log('Found existing model and loaded it');
        return model
    }
    catch (error) {
        console.error('Could Not find model. Creating New One');
        const model = tf.sequential();

        // Add a single input layer
        model.add(tf.layers.dense({ name: 'input-layer', inputShape: [4, 8], units: 50 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ name: 'hidden-1', units: 100, activation: 'tanh' }));
        model.add(tf.layers.dense({ name: 'hidden-2', units: 100, activation: 'tanh' }));
        model.add(tf.layers.dense({ name: 'hidden-3', units: 100, activation: 'tanh' }));

        // Add an output layer
        model.add(tf.layers.dense({ name: 'output-layer', units: 4, activation: 'softmax' }));

        return model;
    }
}

function convertBoardToTensor(board) {
    return tf.tensor3d([board], [1, 4, 8]);
}

function convertToTensor(gameHistory) {
    // Step 1. Shuffle the data
    tf.util.shuffle(gameHistory);

    // Step 2. Convert data to Tensor
    const inputs = gameHistory.map(([state]) => state)
    const labels = gameHistory.map(([, actionIndex,]) => tf.oneHot(actionIndex, ACTIONS_LENGTH).dataSync());
    const rewards = gameHistory.map(([, , rewards]) => rewards);

    let discountedRewards = []
    let discountedSum = 0

    for (reward in rewards.reverse()) {
        discountedSum = reward + DISCOUNT_RATE * discountedSum
        discountedRewards.push(discountedSum)
    }

    const inputTensor = tf.tensor3d(inputs, [inputs.length, inputs[0].length, inputs[0][0].length]);
    const labelTensor = tf.tensor2d(labels, [labels.length, labels[0].length]);
    const rewardsTensor = tf.tensor2d(discountedRewards.reverse(), [rewards.length, 1]);

    return {
        inputs: inputTensor,
        labels: labelTensor,
        rewards: rewardsTensor,
    }
}

async function trainModel(model, gameHistory) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: 'meanSquaredError',
    });

    const batchSize = 1;
    const epochs = 1;

    const { inputs, labels, rewards } = convertToTensor(gameHistory);

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        weights: rewards,
    });
}