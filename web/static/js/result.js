function draw(img_path){
 
    imgClo = new Image();
    imgClo.src = img_path;
    
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
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