navigator.mediaDevices.getUserMedia({
  video: {
    width: 250, 
    height: 250,
    frameRate: 1
  },
  audio: false
}).then(stream => {
  document.getElementById("camera").srcObject = stream; 
}).catch((err) => {
  console.log(err);
}); 

