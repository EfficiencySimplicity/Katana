//https://stackoverflow.com/questions/52059596/loading-an-image-on-web-browser-using-promise
function getCurrentBlendMode() {
    return document.getElementById("blend-mode-select").value;
}

function getCurrentShuffleAmount() {
    return document.getElementById("shuffle-amount-select").value;
}

function getCurrentLayerRatio() {
    return +document.getElementById("ratio-select").value;
}

function clearExistingKatanaBoxes() {
    let existingKatanaBoxes = document.querySelectorAll('.katana-box');
    existingKatanaBoxes.forEach((katanaBox) => katanaBox.remove());
}


// https://stackoverflow.com/questions/35274934/retrieve-image-data-from-file-input-without-a-server
async function createImageBoxFromInputFile() {

    clearExistingKatanaBoxes();

    let images = await imagesFromInputField(document.getElementById('file-input'))
    console.log('Loaded image tensors: ', images);
    images.forEach((image) => createKatanaBoxInDocument(image));

};

async function createKatanaBoxInDocument(image) {
    let t = Date.now();

    tf.tidy(async () => {
        let body = document.getElementById('body');
        let katanaBox = await createKatanaBoxFromImage(
            image,
            getCurrentBlendMode(), 
            getCurrentShuffleAmount(), 
            getCurrentLayerRatio());
        console.log(katanaBox);
        body.appendChild(katanaBox);
    });
    
    image.dispose();

    console.log('Post-operation memory:', tf.memory());
    console.log('Full operation time: ', (Date.now() - t) / 1000);
}
