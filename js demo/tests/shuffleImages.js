
function shuffleLayers(layers) {

    let [h,w,c] = layers[0].shape;
    for (let i=0; i<20; i++) {
        let indexA = Math.floor(Math.random()*layers.length);
        let indexB = Math.floor(Math.random()*(layers.length-1));
        if (indexB >=indexA) {indexB ++;}

        let [a,b] = returnSwapped(layers[indexA], layers[indexB]);

        layers[indexA].dispose();
        layers[indexB].dispose();
        layers[indexA] = a;
        layers[indexB] = b;
    }
    return layers;

}

async function shuffleLayersAsync(layers) {
    layers = await layers;
    let [h,w,c] = layers[0].shape;

    let indices = Array.from(
    { length: layers.length },
    (value, index) => index
    );

    let shufflePromises = [];

    while (indices.length >= 2) {
        let indexA = indices[Math.floor(Math.random()*indices.length)];
        indices.splice(indices.indexOf(indexA), 1)
        let indexB = indices[Math.floor(Math.random()*indices.length)];
        indices.splice(indices.indexOf(indexB), 1)

        shufflePromises.push(new Promise((resolve, reject) => {
            let [a,b] = returnSwapped(layers[indexA], layers[indexB]);
            layers[indexA].dispose();
            layers[indexB].dispose();
            resolve([a,b]);
        }))
    }

    if (indices.length == 1) {
        shufflePromises.push(new Promise((resolve, reject) => {
            resolve([tf.clone(layers[indices[0]])])
        }))
    }

    return await Promise.all(shufflePromises).then(
        (shuffled) => new Promise(
            (resolve, reject) => {
                tf.dispose(layers);
                let unpacked = [];
                shuffled.forEach((shuffleSet) => {
                    shuffleSet.forEach((item) => {
                        unpacked.push(item);
                    })
                })
                resolve(unpacked);
            }
        )
    );
}

// promises here too, or in alike
// MAYBE THE SINGLE BASE LOD LAYER GOTTEN FROM LODS
// IS RESPONSIBLE FOR ALL OF THIS
function returnSwapped(a,b) {return tf.tidy(() => {
    let [h,w,c] = a.shape;

    let sliceX = Math.floor(Math.random()*w/2);
    let sliceY = Math.floor(Math.random()*h/2);
    let sliceH = Math.floor(h/2);
    let sliceW = Math.floor(w/2);

    let sliceA = tf.clone(a.slice([sliceY, sliceX],[sliceH, sliceW]));
    let sliceB = tf.clone(b.slice([sliceY, sliceX],[sliceH, sliceW]));

    let updateMapA = sliceB.pad([[sliceY, h-(sliceY+sliceH)],[sliceX, w-(sliceX+sliceW)],[0,0]],-1);
    let updateMapB = sliceA.pad([[sliceY, h-(sliceY+sliceH)],[sliceX, w-(sliceX+sliceW)],[0,0]],-1);

    return [a.where(tf.equal(updateMapA, -1), updateMapA), b.where(tf.equal(updateMapB, -1), updateMapB)];
})}


image        = tf.randomUniform([150, 321, 3], 0, 1);
padded_image = padImage(image);
LODs         = generateLODsAsync(padded_image);


t = Date.now();

result = shuffleLayersAsync(LODs);
result.then((r) => console.log('Shuffled LODs (async) in ', (Date.now() - t) / 1000, 'Shuffled: ', r));

// t = Date.now();

// result = shuffleLayers(LODs);
// console.log('Shuffled LODs (sync) in ', (Date.now() - t) / 1000, 'Shuffled: ', result)
