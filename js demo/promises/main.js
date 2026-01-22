//https://stackoverflow.com/questions/52059596/loading-an-image-on-web-browser-using-promise

function getCurrentBlendModeSelected() {
    return document.getElementById("blend-mode-select").value;
}

// Gets image array from file input, demo-specific.
// https://stackoverflow.com/questions/35274934/retrieve-image-data-from-file-input-without-a-server
async function createImageBoxFromInputFile() {
    let startTime = Date.now();

    let existingKatanaBox = document.querySelector('.katana-box');
    if (existingKatanaBox) {existingKatanaBox.remove();}

    let input = document.getElementById('file-input');

    let imageBoxPromise = imagesFromInputField(input)
    .then((images) => {
        
        let promises = [];
        images.forEach((image) => {promises.push(
            createKatanaBoxFromImage(image, getCurrentBlendModeSelected())
            .then((img) => {

                let body = document.getElementById('body');
                body.appendChild(img);

            })
        );})

        return Promise.all(promises);
        
    })
    

    await Promise.resolve(imageBoxPromise);

    console.log('Post-operation memory:', tf.memory());
    console.log((Date.now() - startTime) / 1000)
};