

characters = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯',
    'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ',
    'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল',
    'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ']


var base_url = window.location.origin;
let model;
(async function () {
    console.log("model loading...");
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/munem-shahriar697/test1/main/model.json')
    console.log("model loaded..");
})();

function preprocessCanvas(image) {

    //resizing the input image to target size of (1, 28, 28) 
    //tf.browser.fromPixels() method, to create a tensor that will flow into the first layer of the model
    //tf.image.resizeNearestNeighbor() function resizes a batch of 3D images to a new shape
    //tf.mean() function is used to compute the mean of elements across the dimensions of the tensor
    //tf.toFloat() function casts the array to type float
    //The tensor.div() function is used to divide the array or tensor by the maximum RGB value(255)
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat();
    return tensor.div(255.0);
}

//Prediction
//canvas.toDataURL() : returns 
//image in format specified default png
//than send to preprocess function
//await makes program wait until mmodel prediction
//displayLabel to display result
document.getElementById('button_predict').addEventListener("click", async function () {
    var imageData = canvas.toDataURL();
    let tensor = preprocessCanvas(canvas);
    console.log(tensor)
    let predictions = await model.predict(tensor).data();
    let results = Array.from(predictions);
    // displayLabel(results);    
    console.log(results);
    predictShow(results)
});
function predictShow(data) {
    var max = data[0];
    var maxIndex = 0;
    for (var i = 1; i < data.length; i++) {
        if (data[i] > max) {
            maxIndex = i;
            max = data[i];

        }
    }
    console.log(max)
    console.log(maxIndex)
    document.getElementById('result').innerHTML = characters[maxIndex];
    document.getElementById('confidence').innerHTML = "Confidence: " + (max * 100).toFixed(2) + "%";
}
