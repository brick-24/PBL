from flask import Flask, render_template_string, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, vit_b_16
import base64
from io import BytesIO

app = Flask(__name__)


def load_model(model_name):
    if model_name == "resnet":
        model = torch.load("resnet_mnist_full.pth", map_location="cpu", weights_only=False)

    elif model_name == "vgg":
        model = torch.load("vgg_mnist_full.pth", map_location="cpu", weights_only=False)

    elif model_name == "vit":
        from torch.serialization import add_safe_globals
        from timm.models.vision_transformer import VisionTransformer

        add_safe_globals([VisionTransformer])

        model = torch.load("vit_mnist_full.pth", map_location="cpu", weights_only=False)

    else:
        raise ValueError("Invalid model name")

    model.eval()
    return model



transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


page = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial; margin: 40px; background:#fafafa; }
        .container { width: 450px; margin: auto; padding:20px; background:white; border-radius:12px; box-shadow:0 0 10px rgba(0,0,0,0.1);}        
        .dropzone {
            border: 2px dashed #888;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            color: #666;
            cursor: pointer;
            margin-bottom: 15px;
            background:#fdfdfd;
        }
        .dropzone.dragover {
            border-color: #007bff;
            color: #007bff;
        }
        img.preview {
            width: 200px;   /* reduced size */
            margin-top: 15px;
            border-radius: 10px;
            display:none;
        }
        button {
            width:100%; padding:10px; background:#007bff; border:none; color:white;
            border-radius:8px; cursor:pointer; font-size:16px;
        }
    </style>
</head>
<body>
<div class="container">
    <h2>Optical Digit Recognition Using MNIST Dataset</h2>
    <form method="POST" enctype="multipart/form-data">
        <label>Select Model:</label><br>
        <select name="model" style="width:100%; padding:8px; margin-bottom:15px;">
            <option value="resnet">ResNet</option>
            <option value="vgg">VGG</option>
            <option value="vit">Vision Transformer</option>
        </select>

        <div class="dropzone" id="dropzone">Click to Upload</div>
        <input type="file" id="fileInput" name="image" accept="image/*" style="display:none;">

        {% if preview %}
            <img id="preview" class="preview" src="data:image/png;base64,{{ preview }}" style="display:block;" />
        {% else %}
            <img id="preview" class="preview" />
        {% endif %}

        <button type="submit">Predict</button>
    </form>

    {% if result %}
    <div class="result" style="margin-top:20px;">
        <h3>Prediction Result</h3>
        <p><b>Predicted Class:</b> {{ result.pred }}</p>
        <ul>
        {% for i,p in result.probs %}
            <li>Class {{ i }}: {{ p }}%</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    showPreview(file);
});

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    showPreview(file);
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
        preview.src = reader.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}
</script>

</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    preview_base64 = None

    if request.method == 'POST':
        model_name = request.form.get('model')
        file = request.files.get('image')

        # ✅ SELECT TRANSFORM BASED ON MODEL
        if model_name == "vit":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        if file:
            img = Image.open(file.stream).convert('L')

            # base64 for preview
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            preview_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            # transform input
            img_tensor = transform(img).unsqueeze(0)

            # load model
            model = load_model(model_name)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = probs.argmax(dim=1).item()

            result = {
                "pred": predicted_class,
                "probs": [(i, round(float(p.item()*100), 2)) for i, p in enumerate(probs[0])]
            }

    return render_template_string(page, result=result, preview=preview_base64)


if __name__ == '__main__':
    app.run(debug=True)
