

function generateLODs(image, blendmode) {
    let [w,h,c]    = image.shape;
    let num_layers = Math.ceil(Math.log2(h));

    let LODs       = [];

    for (let i=0; i<num_layers; i++) {
        LODs.push(pixellate(image, i+1, blendmode));
    }

    return LODs;
}

async function generateLODsAsync(image, blendmode) {
    let [w,h,c]    = image.shape;
    let num_layers = Math.ceil(Math.log2(h));

    let promises   = [];

    for (let i=0; i<num_layers; i++) {
        promises.push(new Promise((resolve) => resolve(pixellate(image, i+1, blendmode))))
    }

    return await Promise.all(promises);
}


let image        = tf.randomUniform([150, 321, 3], 0, 1);
let padded_image = padImage(image);

let t, result;


t = Date.now();

result = generateLODsAsync(padded_image);
result.then((r) => console.log('Generated LODs (async) in ', (Date.now() - t) / 1000, 'LODs: ', r));

t = Date.now();

result = generateLODs(padded_image);
console.log('Generated LODs (sync) in ', (Date.now() - t) / 1000)

t = Date.now();

result = generateLODsAsync(padded_image);
result.then((r) => console.log('Generated LODs (async) in ', (Date.now() - t) / 1000, 'LODs: ', r));

t = Date.now();

result = generateLODs(padded_image);
console.log('Generated LODs (sync) in ', (Date.now() - t) / 1000)

t = Date.now();

result = generateLODsAsync(padded_image);
result.then((r) => console.log('Generated LODs (async) in ', (Date.now() - t) / 1000, 'LODs: ', r));

t = Date.now();

result = generateLODs(padded_image);
console.log('Generated LODs (sync) in ', (Date.now() - t) / 1000)