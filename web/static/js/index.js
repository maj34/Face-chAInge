function loadFile(input) {
    
    var file = input.files[0];
    var name = document.getElementById('fileName');
    name.textContent = file.name;
    img_path = URL.createObjectURL(file);
    
    draw(img_path)
 
    var button = document.getElementById("button");
    button.style.visibility = "hidden";
};


function draw(img_path){

    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
 
    imgClo = new Image();
    imgClo.src = img_path;


    imgClo.addEventListener('load', function(){
        var w = imgClo.width;  // 1200
        var h = imgClo.height; // 600
        
        var vx = Math.max(w / 1200, h / 600);
        w = w / vx;
        h = h / vx;
        
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(imgClo , 0, 0, canvas.width, canvas.height);
    },false);
}