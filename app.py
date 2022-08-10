from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import PIL
import os
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
class Net(nn.Module):
  def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size = 3),
            nn.ReLU(),
            


            nn.Flatten(),
            nn.Linear(4096,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,3)
            

        )
    
  def forward(self, xb):
        return self.network(xb)




app = Flask(__name__)
@app.route('/')
def image_classifier():
	return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = PIL.Image.open(file_path)
        transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ), transforms.Resize((25, 25)) ] )
        image = transform(image).unsqueeze(0)
        model = torch.load('model.pth')
        model = model.cpu()
        model.eval()
        pred = torch.argmax(model(image))
        print(pred)
        pred=pred.item()
        if pred==0:
         return('This meat is predicted to be Fresh')
        elif pred==1:
         return('This meat is predicted to be HALF-FRESH')
        elif pred== 2:
         return('This meat is predicted to be SPOILT')

        
    return None


    
if __name__ == '__main__':
	app.run()
